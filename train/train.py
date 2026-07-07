#!/usr/bin/env python
"""BitNet healing trainer (train_plan §5/§7; moe_train_plan §5/§6).

The harness shape is adapted from ~/AUM/train/train.py (the template the plans
name in §12): PackedWindows shards, warmup+cosine LR onto snapshotted base lrs,
HealthMonitor with throttled alerts + end-of-run report, save_pretrained
checkpoints with a `latest` symlink and --resume, argparse + YAML configs +
repeatable --set overrides, MPS allocator-cache hygiene. Deltas (each from the
plan): profile-driven converted model, chunked/fused CE+KD via LossComputer,
AdamW(0.9, 0.95) with plan param groups, dual-eval-mode PPL + KL_tf + ternary
panel at every eval, router health + aux loss + cold-expert decay masking for
MoE profiles, and provenance-hash-validated resume (quantizer/config/profile/
manifest hashes + RNG states in trainer_state.pt).

  python train/train.py --config train/configs/a1/t1.yaml
  python train/train.py --config train/configs/ci/tiny.yaml --init <ckpt> --data-dir <shards>
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bitnet_train import provenance  # noqa: E402
from bitnet_train.bitlinear import (  # noqa: E402
    code_flip_rates, iter_bitexperts, set_eval_mode, snapshot_codes, ternary_health)
from bitnet_train.conversion import build_param_groups, load_converted, load_profile  # noqa: E402
from bitnet_train.data import (  # noqa: E402
    IndexedWindows, PackedWindows, calibration_windows, cycle, load_manifest, make_loader)
from bitnet_train.distill import LossComputer, LossConfig, TeacherWrapper, TopkCacheReader  # noqa: E402
from eval_ppl import evaluate_ppl, kl_tf  # noqa: E402


# --------------------------------------------------------------------------- schedule
def lr_scale(step, warmup, total, floor=0.10):
    """Linear warmup then cosine to `floor` of peak (train_plan §5.3)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


# --------------------------------------------------------------------------- health
class HealthMonitor:
    """Rank-0, alert-only (a monitor bug must never kill a long run); throttled
    banners + wandb.alert + report.txt — the AUM skeleton with the §10.4 /
    moe §6.2 rule set. Early steps look like early pretraining and binarized
    loss curves are S-shaped: the loss-rise rule waits out warmup and uses a
    wide margin; nothing here ever kills a run during the plateau."""

    SPIKE_FACTOR, SPIKE_FLOOR = 6.0, 15.0
    LM_WIN, LM_REF, LM_MARGIN = 25, 100, 0.5

    FLIP_FROZEN, FLIP_THRASH = 1e-5, 0.05
    ZERO_DEGENERATE = 0.95

    def __init__(self, say, wandb_on, total_steps, warmup):
        import collections
        self.say, self.wandb_on = say, wandb_on
        self.total_steps, self.warmup = total_steps, warmup
        self.gnorms = collections.deque(maxlen=200)
        self.lms = collections.deque(maxlen=self.LM_REF + self.LM_WIN)
        self.tps = collections.deque(maxlen=50)
        self.spike_steps = collections.deque(maxlen=10)
        self.incidents, self.throttle = [], {}
        self.evals = []
        self.best_val = (float("inf"), None)
        self.mode_gap_hist = collections.deque(maxlen=8)     # a0/a1 divergence trend
        self.entropy_hist = collections.deque(maxlen=8)
        self.maxload_hist = collections.deque(maxlen=8)
        self.dead_hist = collections.deque(maxlen=8)
        self.zerotail_hist = collections.deque(maxlen=8)
        self.t_start = time.time()
        self.start_step = None

    def _alert(self, step, level, kind, msg, every=100):
        if step - self.throttle.get(kind, -10**9) < every:
            return
        self.throttle[kind] = step
        self.incidents.append((step, level, msg))
        bar = "!" * 78
        self.say(f"\n{bar}\n!!! {level} @ step {step}: {msg}\n{bar}")
        if self.wandb_on:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.alert(title=f"bitnet {level}: {kind}", text=f"step {step}: {msg}")
            except Exception:
                pass

    def on_step(self, step, metrics, tps):
        if self.start_step is None:
            self.start_step = step
        loss, gn = metrics.get("loss"), metrics.get("grad_norm")
        if loss is not None and not math.isfinite(loss):
            self._alert(step, "CRITICAL", "nan_loss", f"non-finite loss ({loss})", every=1)
        if gn is not None:
            if not math.isfinite(gn):
                self._alert(step, "CRITICAL", "nan_grad", f"non-finite grad norm ({gn})",
                            every=1)
            elif len(self.gnorms) >= 50:
                med = sorted(self.gnorms)[len(self.gnorms) // 2]
                if gn > max(self.SPIKE_FACTOR * med, self.SPIKE_FLOOR):
                    self.spike_steps.append(step)
                    self._alert(step, "WARN", "grad_spike",
                                f"grad-norm spike {gn:.1f} (recent median {med:.2f})",
                                every=20)
                    if len(self.spike_steps) >= 3 and step - self.spike_steps[-3] <= 50:
                        self._alert(step, "CRITICAL", "grad_unstable",
                                    "3+ grad-norm spikes within 50 steps — sustained "
                                    "numerical instability")
            if math.isfinite(gn):
                self.gnorms.append(gn)
        ce = metrics.get("ce")
        if ce is not None and math.isfinite(ce) and step > 2 * self.warmup:
            self.lms.append(ce)                          # S-curve tolerance: post-warmup only
            if len(self.lms) == self.lms.maxlen:
                recent = sum(list(self.lms)[-self.LM_WIN:]) / self.LM_WIN
                ref = sum(list(self.lms)[:self.LM_REF]) / self.LM_REF
                if recent > ref + self.LM_MARGIN:
                    self._alert(step, "WARN", "lm_rising",
                                f"CE rising post-warmup: last-{self.LM_WIN} mean "
                                f"{recent:.3f} vs prior-{self.LM_REF} mean {ref:.3f}")
        if tps and math.isfinite(tps):
            if len(self.tps) >= 20:
                med = sorted(self.tps)[len(self.tps) // 2]
                if tps < 0.5 * med:
                    self._alert(step, "WARN", "throughput",
                                f"throughput sag: {tps / 1e3:.1f}k tok/s vs median "
                                f"{med / 1e3:.1f}k")
            self.tps.append(tps)

    def on_eval(self, step, ev: dict):
        val = ev.get("val_ce_primary")
        if val is not None:
            self.evals.append((step, val))
            if val < self.best_val[0]:
                self.best_val = (val, step)
        flip = ev.get("flip_total")
        if flip is not None:
            if step > 2 * self.warmup and flip < self.FLIP_FROZEN:
                self._alert(step, "WARN", "flip_frozen",
                            f"code-flip rate {flip:.2e} ~ zero: latents drift without "
                            "crossing thresholds — the effective model is NOT training "
                            "(the low-LR failure, train_plan §10.2)", every=200)
            elif step > 4 * self.warmup and flip > self.FLIP_THRASH:
                self._alert(step, "WARN", "flip_thrash",
                            f"code-flip rate {flip:.3f} sustained high — thrashing "
                            "(the high-LR failure)", every=200)
        zmax = ev.get("zero_frac_max")
        if zmax is not None and zmax > self.ZERO_DEGENERATE:
            self._alert(step, "WARN", "ternary_degenerate",
                        f"a layer's zero-code fraction is {zmax:.2f} — ternary "
                        "degeneracy (§10.2)", every=200)
        gap = ev.get("mode_gap")
        if gap is not None:
            self.mode_gap_hist.append(gap)
            h = list(self.mode_gap_hist)
            if len(h) >= 4 and all(b > a for a, b in zip(h[-4:], h[-3:])) \
                    and h[-1] - h[-4] > 0.05:
                self._alert(step, "WARN", "mode_divergence",
                            f"a0/a1 (W-only vs W2A8) CE gap TREND rising "
                            f"({h[-4]:.3f} -> {h[-1]:.3f}): validation and deployment "
                            "objectives are separating (moe §6.3)", every=200)
        ent, ml = ev.get("router/entropy_mean"), ev.get("router/max_load_mean")
        if ent is not None and ml is not None:
            self.entropy_hist.append(ent)
            self.maxload_hist.append(ml)
            e, m = list(self.entropy_hist), list(self.maxload_hist)
            if len(e) >= 4 and all(b < a for a, b in zip(e[-4:], e[-3:])) \
                    and all(b > a for a, b in zip(m[-4:], m[-3:])):
                self._alert(step, "WARN", "router_collapse",
                            f"routing entropy declining ({e[-4]:.3f}->{e[-1]:.3f}) with "
                            f"max-load rising ({m[-4]:.3f}->{m[-1]:.3f}) — the collapse "
                            "canary; raise beta_aux one step (moe §6.2)", every=200)
        dead = ev.get("router/dead_experts")
        if dead is not None:
            self.dead_hist.append(dead)
            if len(self.dead_hist) >= 3 and min(list(self.dead_hist)[-3:]) > 0:
                self._alert(step, "WARN", "dead_experts",
                            f"{dead} experts below utilization eps for 3+ evals",
                            every=200)
        zt = ev.get("router/zero_code_cold_tail")
        zh = ev.get("router/zero_code_hot", 0.0)
        if zt is not None:
            self.zerotail_hist.append(zt)
            h = list(self.zerotail_hist)
            if len(h) >= 3 and h[-1] > h[-3] and zt > zh + 0.1:
                self._alert(step, "WARN", "decay_erosion",
                            f"zero-code fraction rising in the LOW-utilization tail "
                            f"({zt:.3f} vs hot {zh:.3f}) — VERIFY THE DECAY MASK IS "
                            "ACTIVE before touching anything else (moe §6.2)", every=200)

    def report(self, run_dir, final_step, tokens_seen):
        hrs = (time.time() - self.t_start) / 3600
        lines = ["=" * 78, "BITNET HEAL — TRAINING HEALTH REPORT", "=" * 78,
                 f"steps {self.start_step or '?'}..{final_step} of {self.total_steps} | "
                 f"{tokens_seen:,} tokens | {hrs:.2f} h wall"]
        if self.evals:
            v0, vN = self.evals[0], self.evals[-1]
            lines.append(f"val CE: first {v0[1]:.4f} @ {v0[0]} | best "
                         f"{self.best_val[0]:.4f} @ {self.best_val[1]} | final "
                         f"{vN[1]:.4f} @ {vN[0]}")
        if self.gnorms:
            gs = sorted(self.gnorms)
            lines.append(f"grad norm: median {gs[len(gs) // 2]:.2f} "
                         f"p95 {gs[int(0.95 * (len(gs) - 1))]:.2f} max {gs[-1]:.2f}")
        crit = [i for i in self.incidents if i[1] == "CRITICAL"]
        warn = [i for i in self.incidents if i[1] == "WARN"]
        lines.append(f"incidents: {len(crit)} CRITICAL, {len(warn)} WARN"
                     + ("" if self.incidents else " — clean run"))
        for step, level, msg in self.incidents[-30:]:
            lines.append(f"  [{level}] step {step}: {msg.splitlines()[0]}")
        lines.append("=" * 78)
        text = "\n".join(lines)
        self.say(text)
        try:
            (Path(run_dir) / "report.txt").write_text(text + "\n")
        except OSError:
            pass
        return text


# --------------------------------------------------------------------------- checkpointing
def save_checkpoint(accelerator, model, optimizer, run_dir, step, tokens_seen, meta,
                    seeds):
    path = os.path.join(run_dir, f"step-{step:06d}")
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path)
    accelerator.wait_for_everyone()
    state = {"optimizer": optimizer.state_dict(), "step": step,
             "tokens_seen": tokens_seen,
             "meta": {**meta, "seeds": seeds,
                      "rng_states": provenance.rng_capture()}}
    name = "trainer_state.pt" if accelerator.is_main_process \
        else f"trainer_state_rank{accelerator.process_index}.pt"
    torch.save(state, os.path.join(path, name))
    if accelerator.is_main_process:
        latest = os.path.join(run_dir, "latest")
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    accelerator.wait_for_everyone()
    return path


def validate_resume_meta(saved: dict, current: dict, allow_mismatch: bool, say):
    """Re-derive every derivable hash and hard-fail on drift (§5.6): a changed
    quantizer, profile, config, or corpus silently invalidates comparability."""
    bad = []
    for k in ("quantizer_hash", "profile_hash", "config_hash", "manifest_hash"):
        if k in saved and k in current and saved[k] != current[k]:
            bad.append(f"{k}: checkpoint {saved[k]} != current {current[k]}")
    if bad and not allow_mismatch:
        raise SystemExit("[resume] provenance mismatch (pass --allow-hash-mismatch "
                         "to override, logged as an incident):\n  " + "\n  ".join(bad))
    for b in bad:
        say(f"WARNING: resume hash mismatch (overridden): {b}")
    return bad


# --------------------------------------------------------------------------- main
def build_argparser(here: Path) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BitNet healing trainer")
    ap.add_argument("--config", default=None, help="YAML with defaults for any flag")
    ap.add_argument("--profile", default=str(here / "profiles" / "a1.yaml"))
    ap.add_argument("--init", required=False, default=None,
                    help="converted checkpoint (train/init_from_base.py output)")
    ap.add_argument("--resume", default=None, help="step-NNNNNN dir to resume from")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--run-name", default="bitnet-heal")
    ap.add_argument("--out-dir", default=str(here / "checkpoints"))
    ap.add_argument("--backend", default=None, choices=[None, "reference", "metal"],
                    help="BitLinear backend (default: metal on MPS, else reference)")
    # recipe (train_plan §5.6 starter values live in configs/a1/t1.yaml)
    ap.add_argument("--total-tokens", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak lr (T1 grid §5.3)")
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="2-D BitLinear latents only (§5.2)")
    ap.add_argument("--expert-weight-decay", type=float, default=0.0,
                    help="Q-track expert decay, applied MASKED. Stays 0 until "
                         "tests/test_decay_mask.py is green under the real wrapping "
                         "(moe §5.2 fallback ordering)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lambda-ramp-steps", type=int, default=0,
                    help="A3/Q-A3: ramp quantization in over N steps (0 = off, lam=1 "
                         "from step 0). Enable only on step-0 spikes (§9.1)")
    ap.add_argument("--mixed-precision", default="no", choices=["no", "bf16", "fp16"])
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--latent-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="§5.4: fp32 baseline (A1) or bf16 latents + fp32 masters (hatch)")
    ap.add_argument("--moments-bits", type=int, default=32, choices=[32, 8],
                    help="§5.4: 8 = blockwise-quantized Adam moments (memory hatch)")
    # distillation (§5.1)
    ap.add_argument("--kd", default="dense", choices=["none", "dense", "topk"])
    ap.add_argument("--kd-alpha", type=float, default=1.0)
    ap.add_argument("--kd-tau", type=float, default=2.0)
    ap.add_argument("--kd-tail-mode", type=int, default=0, choices=[0, 1])
    ap.add_argument("--topk-cache", default=None, help="dir from distill.build_topk_cache")
    ap.add_argument("--teacher", default=None, help="override profile.teacher")
    ap.add_argument("--vchunk", type=int, default=8192)
    ap.add_argument("--tchunk", type=int, default=1024)
    ap.add_argument("--no-fused-losses", action="store_true")
    # cadence
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--calib-windows", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--empty-cache-every", type=int, default=None)
    ap.add_argument("--no-tqdm", action="store_true")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--wandb-project", default="bitnet-heal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any parsed arg, applied last (e.g. --set lr=2e-4)")
    return ap


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    ap = build_argparser(here)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        import yaml
        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        dests = {a.dest for a in ap._actions}
        unknown = set(cfg) - dests
        if unknown:
            raise SystemExit(f"--config: unknown keys {sorted(unknown)}")
        ap.set_defaults(**cfg)
    args = ap.parse_args(argv)
    dests = {a.dest: a for a in ap._actions}
    for kv in args.set:
        key, eq, val = kv.partition("=")
        key = key.replace("-", "_")
        if not eq or key not in dests:
            raise SystemExit(f"--set: {kv!r} is not KEY=VALUE over a known flag")
        try:
            setattr(args, key, ast.literal_eval(val))
        except (ValueError, SyntaxError):
            setattr(args, key, val)
    if args.empty_cache_every is None:
        args.empty_cache_every = 10 if torch.backends.mps.is_available() else 0
    if args.wandb is None:
        try:
            import wandb  # noqa: F401
            args.wandb = True
        except ImportError:
            args.wandb = False
    return args


def main(argv=None):
    args = parse_args(argv)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision=args.mixed_precision,
                              log_with="wandb" if args.wandb else None,
                              kwargs_handlers=[DistributedDataParallelKwargs(
                                  find_unused_parameters=False)])   # §5.6 (A-track);
    # Q-track trains under FSDP where the flag is moot; DDP debug configs set it True.
    set_seed(args.seed)

    profile = load_profile(args.profile)
    backend = args.backend or ("metal" if torch.backends.mps.is_available()
                               else "reference")
    if args.init is None:
        raise SystemExit("--init is required (run train/init_from_base.py first)")
    ckpt = args.resume or args.init
    latent_dtype = torch.bfloat16 if args.latent_dtype == "bf16" else torch.float32
    model, conv_report = load_converted(ckpt, profile, backend=backend,
                                        dtype=latent_dtype)      # §5.4: fp32 baseline / bf16 hatch
    if args.grad_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except (AttributeError, ValueError) as e:
            accelerator.print(f"WARNING: gradient checkpointing unavailable ({e})")
    n_params = sum(p.numel() for p in model.parameters())

    # ---- MoE plumbing (profile-gated)
    router_hooks = masker = None
    teacher_routing = None
    is_moe = bool(profile.expert_stack_regexes) and any(iter_bitexperts(model))
    if is_moe:
        from bitnet_train.moe_metrics import RouterHooks
        router_hooks = RouterHooks(model).attach()
        router_hooks.collect_aux = profile.aux_loss

    # ---- data
    train_ds = PackedWindows(args.data_dir, "train", args.seq_len)
    train_loader = make_loader(IndexedWindows(train_ds), args.batch_size, args.seed)
    calib = calibration_windows(args.data_dir, args.calib_windows,
                                split="val" if "val" in train_ds.manifest["splits"]
                                else "train", seq_len=args.seq_len)

    tokens_per_step = (args.batch_size * args.grad_accum * args.seq_len
                       * accelerator.num_processes)
    total_tokens = args.total_tokens if args.total_tokens is not None \
        else len(train_ds) * args.seq_len * args.epochs
    total_steps = max(1, math.ceil(total_tokens / tokens_per_step))

    # ---- optimizer (§5.2): AdamW(0.9, 0.95); wd on 2-D latents only; Q groups via profile
    groups = build_param_groups(model, profile, args.lr, weight_decay=args.weight_decay)
    for g in groups:
        if g.get("decay_masked"):
            g["weight_decay"] = 0.0            # masker owns expert decay (§5.2 ordering)
    if args.latent_dtype == "bf16" or args.moments_bits == 8:
        from bitnet_train.optim import MasterAdamW
        optimizer = MasterAdamW(groups, betas=(0.9, 0.95), eps=1e-8,
                                moments_bits=args.moments_bits)   # fp32 masters + opt hatches
    else:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    if is_moe and profile.decay_masking:
        from bitnet_train.optim import ColdExpertDecayMasker
        masker = ColdExpertDecayMasker(model, optimizer, args.expert_weight_decay)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    raw_model = accelerator.unwrap_model(model)

    # ---- teacher / KD (§5.1: KD by default; live dense teacher for A-track)
    teacher = cache_reader = None
    loss_cfg = LossConfig(alpha=args.kd_alpha, tau=args.kd_tau, kd_mode=args.kd,
                          tail_mode=args.kd_tail_mode, vchunk=args.vchunk,
                          tchunk=args.tchunk, prefer_fused=not args.no_fused_losses)
    loss_computer = LossComputer(loss_cfg)
    teacher_id = args.teacher or profile.teacher
    if args.kd == "dense":
        if teacher_id == "__self__":
            teacher = TeacherWrapper(args.init, accelerator.device)   # the dense twin
        else:
            teacher = TeacherWrapper(teacher_id, accelerator.device)
    elif args.kd == "topk":
        if not args.topk_cache:
            raise SystemExit("--kd topk requires --topk-cache (distill.build_topk_cache)")
        cache_reader = TopkCacheReader(args.topk_cache, args.data_dir, tau=args.kd_tau)

    # ---- provenance meta (§5.6, binding)
    meta = provenance.build_meta(
        profile_path=args.profile, model_config=raw_model.config,
        data_manifest=load_manifest(args.data_dir),
        extra={"backend": backend, "kd": args.kd, "init": str(args.init),
               "args": {k: v for k, v in vars(args).items() if k != "set"},
               "teacher_cache": args.topk_cache})
    start_step, tokens_seen = 0, 0
    if args.resume:
        st_path = os.path.join(args.resume,
                               f"trainer_state_rank{accelerator.process_index}.pt")
        if accelerator.process_index == 0 or not os.path.exists(st_path):
            st_path = os.path.join(args.resume, "trainer_state.pt")
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        validate_resume_meta(st.get("meta", {}), meta, args.allow_hash_mismatch,
                             accelerator.print)
        optimizer.load_state_dict(st["optimizer"])
        start_step, tokens_seen = st["step"], st["tokens_seen"]
        if "rng_states" in st.get("meta", {}):
            provenance.rng_restore(st["meta"]["rng_states"])

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "metrics.jsonl")
    if args.wandb:
        try:
            accelerator.init_trackers(args.wandb_project,
                                      config={k: v for k, v in vars(args).items()
                                              if k != "set"},
                                      init_kwargs={"wandb": {"name": args.run_name,
                                                             "resume": "allow"}})
        except Exception as e:
            accelerator.print(f"WARNING: wandb init failed ({e}); continuing without")
            args.wandb = False

    accelerator.print(
        f"bitnet heal [{profile.name}]: {n_params / 1e6:.1f}M params "
        f"({conv_report.n_ternarized} BitLinear + {conv_report.n_expert_stacks} expert "
        f"stacks, ternary frac {conv_report.ternary_param_fraction:.2f}) on "
        f"{accelerator.device} backend={backend}\n"
        f"  {tokens_per_step:,} tok/step x {total_steps:,} steps = "
        f"{tokens_per_step * total_steps / 1e9:.3f}B tokens | kd={args.kd} "
        f"alpha={args.kd_alpha} tau={args.kd_tau} | lr={args.lr} warmup={args.warmup_steps}\n"
        f"  manifest {meta['manifest_hash']} quantizer {meta['quantizer_hash']} "
        f"profile {meta.get('profile_hash', '?')}"
        + (f"\n  resumed from {args.resume} @ step {start_step}" if args.resume else ""))

    from tqdm.auto import tqdm
    bar = tqdm(total=total_steps, initial=start_step, unit="step", dynamic_ncols=True,
               disable=args.no_tqdm or not accelerator.is_main_process,
               desc=args.run_name)
    say = bar.write if not bar.disable else accelerator.print
    health = HealthMonitor(say, args.wandb, total_steps, args.warmup_steps) \
        if accelerator.is_main_process else None

    train_iter = cycle(train_loader)
    head_w = raw_model.get_output_embeddings().weight
    train_mode = "a1" if is_moe else "w_a8"
    aux_coef = float(getattr(raw_model.config, "router_aux_loss_coef", 0.0) or 0.0)
    if is_moe and teacher is not None and accelerator.is_main_process:
        from bitnet_train.moe_metrics import capture_teacher_routing
        teacher_routing = capture_teacher_routing(
            teacher.model, calib.to(accelerator.device), accelerator.device,
            raw_model.config.num_experts_per_tok)     # Q-T0 §8.1 fixture
    code_snapshot = snapshot_codes(raw_model) if accelerator.is_main_process else None
    t0, tokens0 = time.time(), tokens_seen
    model.train()

    def one_micro_step(batch):
        widx, ids = batch
        out = model(ids, output_hidden_states=True, logits_to_keep=1)
        hidden = out.hidden_states[-1][:, :-1, :]            # post-final-norm (B,T-1,H)
        flat_h = hidden.reshape(-1, hidden.shape[-1])
        targets = ids[:, 1:].reshape(-1)
        teacher_batch = None
        if args.kd == "dense":
            teacher_batch = teacher.slice_server(ids)
        elif args.kd == "topk":
            t_idx, t_prob = cache_reader.batch(widx, ids.device)
            teacher_batch = (t_idx[:, :-1].reshape(-1, t_idx.shape[-1]),
                             t_prob[:, :-1].reshape(-1, t_prob.shape[-1]))
        parts = loss_computer(flat_h, head_w, targets, teacher_batch)
        loss = parts["loss"]
        aux_val = 0.0
        if router_hooks is not None and profile.aux_loss and aux_coef > 0:
            aux = router_hooks.aux_loss(raw_model.config.num_experts,
                                        raw_model.config.num_experts_per_tok, aux_coef)
            loss = loss + aux.to(loss.device)
            aux_val = float(aux.detach())
        elif router_hooks is not None:
            router_hooks.clear_aux()
        accelerator.backward(loss)
        return {"loss": float(loss.detach()), "ce": float(parts["ce"]),
                "kd": float(parts["kd"]), "aux": aux_val}

    from bitnet_train.bitlinear import set_lambda

    for step in range(start_step, total_steps):
        scale = lr_scale(step, args.warmup_steps, total_steps)
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale
        if args.lambda_ramp_steps:
            set_lambda(raw_model, (step + 1) / args.lambda_ramp_steps)

        step_metrics = None
        for _ in range(args.grad_accum):
            with accelerator.accumulate(model):
                step_metrics = one_micro_step(next(train_iter))
                if accelerator.sync_gradients:
                    gn = accelerator.clip_grad_norm_(model.parameters(),
                                                     args.max_grad_norm)
                    step_metrics["grad_norm"] = float(gn)
                optimizer.step()
                if accelerator.sync_gradients and masker is not None:
                    masker.step(router_hooks.routed_and_reset())
                optimizer.zero_grad()
        tokens_seen += tokens_per_step

        if args.empty_cache_every and (step + 1) % args.empty_cache_every == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        if health:
            try:
                dt = time.time() - t0
                health.on_step(step + 1, step_metrics,
                               (tokens_seen - tokens0) / dt if dt > 0 else None)
            except Exception as e:
                say(f"WARNING: health monitor error ({e}); continuing")
        bar.update(1)
        bar.set_postfix(loss=f"{step_metrics['loss']:.3f}", lr=f"x{scale:.2f}",
                        refresh=False)

        if (step + 1) % args.log_every == 0 and accelerator.is_main_process:
            dt = time.time() - t0
            tps = (tokens_seen - tokens0) / max(dt, 1e-9)
            gn = step_metrics.get("grad_norm")
            say(f"step {step + 1:>6}/{total_steps} loss {step_metrics['loss']:.4f} "
                f"[ce {step_metrics['ce']:.4f} kd {step_metrics['kd']:.4f}"
                + (f" aux {step_metrics['aux']:.4f}" if is_moe else "") + "] "
                + (f"gnorm {gn:.2f} " if gn is not None else "")
                + f"lr x{scale:.3f} {tps / 1e3:.1f}k tok/s")
            with open(log_path, "a") as f:
                json.dump({"step": step + 1, "tokens": tokens_seen,
                           "lr_scale": scale, **step_metrics}, f)
                f.write("\n")
            if args.wandb:
                accelerator.log({"train/loss": step_metrics["loss"],
                                 "train/ce": step_metrics["ce"],
                                 "train/kd": step_metrics["kd"],
                                 "train/aux": step_metrics["aux"],
                                 **({"train/grad_norm": gn} if gn is not None else {}),
                                 "lr_scale": scale, "tokens": tokens_seen,
                                 "tokens_per_s": tps}, step=step + 1)
            t0, tokens0 = time.time(), tokens_seen

        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            ev = run_eval(model, raw_model, profile, calib, accelerator, teacher,
                          args, is_moe, router_hooks, train_mode, teacher_routing)
            if accelerator.is_main_process:
                new_snap = snapshot_codes(raw_model)
                flips = code_flip_rates(code_snapshot, new_snap)
                code_snapshot = new_snap
                ev["flip_total"] = flips["_total"]
                th = ternary_health(raw_model)
                ev["zero_frac_mean"] = sum(v["frac_zero"] for v in th.values()) / len(th)
                ev["zero_frac_max"] = max(v["frac_zero"] for v in th.values())
                ev["quant_rel_err_mean"] = (sum(v["quant_rel_err"] for v in th.values())
                                            / len(th))
                say(f"step {step + 1:>6}: "
                    + " ".join(f"{k}={v:.4f}" for k, v in ev.items()
                               if isinstance(v, float)))
                if health:
                    try:
                        health.on_eval(step + 1, ev)
                    except Exception as e:
                        say(f"WARNING: health monitor error ({e}); continuing")
                with open(log_path, "a") as f:
                    json.dump({"step": step + 1,
                               **{k: v for k, v in ev.items()
                                  if isinstance(v, (int, float))}}, f)
                    f.write("\n")
                if args.wandb:
                    accelerator.log({f"val/{k}": v for k, v in ev.items()
                                     if isinstance(v, (int, float))}, step=step + 1)
            model.train()

        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            path = save_checkpoint(accelerator, model, optimizer, run_dir, step + 1,
                                   tokens_seen, meta, seeds={"seed": args.seed})
            if accelerator.is_main_process:
                say(f"saved {path}")

    bar.close()
    accelerator.print(f"done: {tokens_seen:,} tokens")
    if health:
        try:
            health.report(run_dir, total_steps, tokens_seen)
        except Exception as e:
            accelerator.print(f"WARNING: health report failed ({e})")
    if args.wandb:
        accelerator.end_training()
    return run_dir


@torch.no_grad()
def run_eval(model, raw_model, profile, calib, accelerator, teacher, args, is_moe,
             router_hooks, train_mode, teacher_routing=None):
    """Both eval modes + KL_tf on the FIXED calibration windows (§8.4/§6.3: both
    modes at every eval, alarm on trend) + the router panel + per-layer top-8
    teacher-routing agreement (moe §6.2)."""
    model.eval()
    device = accelerator.device
    ev = {}
    ces = {}
    if router_hooks is not None:
        router_stats = router_hooks.stats_and_reset(raw_model)   # training-interval stats
        ev.update({k: v for k, v in router_stats.items() if isinstance(v, (int, float))})
        router_hooks.collect_aux = False
    for mode in profile.eval_modes:
        r = evaluate_ppl(raw_model, calib, device, mode=mode)
        ces[mode] = r["ce"]
        ev[f"ce_{mode}"] = r["ce"]
        ev[f"ppl_{mode}"] = r["ppl"]
    if len(profile.eval_modes) >= 2:
        m0, m1 = profile.eval_modes[0], profile.eval_modes[1]
        ev["mode_gap"] = ces[m0] - ces[m1]                       # a0-a1 / w_only-w_a8
    ev["val_ce_primary"] = ces[profile.eval_modes[0]]
    if teacher is not None:
        set_eval_mode(raw_model, train_mode)
        ev["kl_tf"] = kl_tf(raw_model, teacher.model, calib, device, tau=1.0)
    set_eval_mode(raw_model, train_mode)
    if router_hooks is not None:
        router_hooks.stats_and_reset()                            # drop eval-pass stats
        if teacher_routing is not None:                           # §6.2 agreement pass
            router_hooks.capture_routing = True
            raw_model(calib.to(device))
            router_hooks.capture_routing = False
            agree = router_hooks.agreement_vs_teacher(teacher_routing)
            ev["router/top8_agreement"] = agree["_mean"]
            for i, a in enumerate(agree["_by_depth"]):
                ev[f"router/agree_L{i}"] = a
            router_hooks.stats_and_reset()
        router_hooks.collect_aux = profile.aux_loss
    return ev


if __name__ == "__main__":
    main()
