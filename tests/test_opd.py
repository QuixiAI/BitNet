"""OPD estimators + loop (train_plan §6.3, §6.3.1). Built now per an explicit
build-everything override of the §6.2 pre-T2 ban; correctness is testable
independent of the milestone gate."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_train.opd import estimators, rollout_metrics  # noqa: E402

torch.manual_seed(0)
T, H, V = 16, 32, 128


def _rig():
    hidden = torch.randn(T, H, requires_grad=True)
    head_w = (torch.randn(V, H) * 0.1)
    teacher_logits = torch.randn(T, V) * 1.5
    teacher_logp = F.log_softmax(teacher_logits, -1)
    return hidden, head_w, teacher_logp


def test_e0_matches_manual_reverse_kl():
    hidden, head_w, teacher_logp = _rig()
    loss = estimators.full_reverse_kl(hidden, head_w,
                                      lambda idx: teacher_logp.index_select(1, idx),
                                      vchunk=64)
    logp_s = F.log_softmax(hidden.float() @ head_w.t(), -1)
    ref = (logp_s.exp() * (logp_s - teacher_logp)).sum(-1).mean()
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-5)
    loss.backward()                                     # differentiable through student
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_e3_equals_e0_when_support_is_full_vocab():
    """E3 with a full-vocab support and renorm must reproduce E0 (the support
    truncation is the only approximation)."""
    hidden, head_w, teacher_logp = _rig()
    full_support = torch.arange(V).unsqueeze(0).expand(T, V)
    e3 = estimators.support_reverse_kl(
        hidden, head_w, full_support,
        lambda idx: teacher_logp.gather(1, idx), tail_mode="renorm", vchunk=64)
    e0 = estimators.full_reverse_kl(hidden, head_w,
                                    lambda idx: teacher_logp.index_select(1, idx),
                                    vchunk=64)
    torch.testing.assert_close(e3, e0, rtol=1e-4, atol=1e-5)


def test_e3_support_contains_sampled_and_student_topk():
    """§6.3.1: the E3 support must include the sampled token and the student's
    own top-k — the tokens a teacher-top-k-only support (E2) misses."""
    hidden, head_w, teacher_logp = _rig()
    teacher_topk = teacher_logp.topk(8, -1).indices
    sampled = torch.randint(0, V, (T,))
    support = estimators.build_e3_support(hidden, head_w, teacher_topk, sampled,
                                          student_k=8)
    with torch.no_grad():
        student_top = (hidden @ head_w.t()).topk(8, -1).indices
    for t in range(T):
        s = set(support[t].tolist())
        assert sampled[t].item() in s
        assert set(student_top[t].tolist()) <= s
        assert set(teacher_topk[t].tolist()) <= s


def test_e2_biased_away_from_student_error_tokens():
    """The plan's headline: E2 (teacher-top-k support) systematically differs
    from E3 when the student concentrates mass off the teacher's top-k — exactly
    the error tokens OPD must fix. Construct such a case and show the gradients
    differ."""
    hidden = torch.randn(T, H, requires_grad=True)
    head_w = torch.randn(V, H) * 0.1
    # teacher concentrated on [0..7]; force the student to prefer high indices
    teacher_logits = torch.full((T, V), -10.0)
    teacher_logits[:, :8] = torch.randn(T, 8)
    teacher_logp = F.log_softmax(teacher_logits, -1)
    with torch.no_grad():
        head_w[100:108] += 3.0                          # push student mass to 100..107
    teacher_topk = teacher_logp.topk(16, -1).indices
    sampled = torch.full((T,), 100)

    e3_support = estimators.build_e3_support(hidden, head_w, teacher_topk, sampled)
    g_e3 = torch.autograd.grad(
        estimators.support_reverse_kl(hidden, head_w, e3_support,
                                      lambda idx: teacher_logp.gather(1, idx)),
        hidden, retain_graph=True)[0]
    g_e2 = torch.autograd.grad(
        estimators.teacher_topk_reverse_kl(hidden, head_w, teacher_topk,
                                           lambda idx: teacher_logp.gather(1, idx)),
        hidden)[0]
    assert (g_e3 - g_e2).norm() / g_e3.norm().clamp_min(1e-9) > 0.1


def test_tail_modes_differ():
    hidden, head_w, teacher_logp = _rig()
    support = teacher_logp.topk(8, -1).indices
    r = estimators.support_reverse_kl(hidden, head_w, support,
                                      lambda idx: teacher_logp.gather(1, idx),
                                      tail_mode="renorm")
    o = estimators.support_reverse_kl(hidden, head_w, support,
                                      lambda idx: teacher_logp.gather(1, idx),
                                      tail_mode="other")
    assert abs(float(r) - float(o)) > 1e-4


def test_rollout_metrics():
    lp = F.log_softmax(torch.randn(20, V), -1)
    assert rollout_metrics.rollout_entropy(lp) > 0
    toks = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 7]]
    assert 0 < rollout_metrics.distinct_n(toks, 2) <= 1
    assert rollout_metrics.repetition_rate([[1, 1, 1, 1]], 2) > 0
    assert rollout_metrics.early_eos_rate([2, 3, 50], 100) == pytest.approx(2 / 3)
    s = rollout_metrics.summarize(0.5, 0.2, 3.0, toks, [6, 4], 64)
    assert s["on_policy_gap"] == pytest.approx(0.3)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("transformers"),
                    reason="needs transformers")
def test_opd_step_smoke():
    from transformers import LlamaConfig, LlamaForCausalLM
    from bitnet_train.conversion import convert, load_profile
    from bitnet_train.distill import TeacherWrapper
    from bitnet_train.opd.gkd_loop import OPDConfig, opd_step

    torch.manual_seed(0)
    cfg_m = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, vocab_size=256,
                        tie_word_embeddings=True, eos_token_id=0)
    student = LlamaForCausalLM(cfg_m)
    prof = load_profile(Path(__file__).resolve().parents[1] / "train" / "profiles"
                        / "ci_tiny.yaml")
    convert(student, prof, backend="reference")
    teacher = TeacherWrapper.__new__(TeacherWrapper)
    torch.nn.Module.__init__(teacher)
    teacher.model = LlamaForCausalLM(cfg_m).eval().requires_grad_(False)

    prompt = torch.randint(1, 256, (1, 8))
    cfg = OPDConfig(estimator="e3", max_new_tokens=6, vchunk=64)
    loss, m = opd_step(student, teacher, prompt, cfg, on_policy=True)
    assert torch.isfinite(loss) and m["gen_tokens"] > 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters() if p.requires_grad)
