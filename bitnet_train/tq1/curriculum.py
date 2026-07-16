"""Token-domain, resumable soft/hard/frozen TQ1 QAT curriculum gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .qat import TQ1Linear


_LEGACY_STEP_KEYS = {
    "soft_steps", "hard_steps", "freeze_indices_at", "freeze_max_step",
}


def schedule_from_config(config: Mapping[str, Any]) -> "QATSchedule":
    legacy = sorted(_LEGACY_STEP_KEYS & set(config))
    if legacy:
        raise ValueError(
            "legacy step-domain QAT schedule keys are unsupported: " + ", ".join(legacy))
    required = {"soft_tokens", "hard_tokens", "freeze_eval_every_tokens"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"token-domain QAT schedule is missing {missing}")
    return QATSchedule(
        initial_phase=str(config["qat_projection"]),
        temperature_start=float(config.get("temperature_start", 1.0)),
        temperature_end=float(config.get("temperature_end", 0.05)),
        soft_tokens=config["soft_tokens"],
        hard_tokens=config["hard_tokens"],
        freeze_eval_every_tokens=config["freeze_eval_every_tokens"],
        freeze_indices_at_tokens=config.get("freeze_indices_at_tokens"),
        freeze_max_tokens=config.get("freeze_max_tokens"),
        flip_threshold=float(config.get("freeze_flip_threshold", 1e-4)),
        margin_threshold=float(config.get("freeze_margin_threshold", 0.0)),
        sustain_evals=config.get("freeze_sustain_evals", 3),
        trend_tolerance=float(config.get("freeze_trend_tolerance", 0.01)),
        max_zero_fraction=float(config.get("freeze_max_zero_fraction", 0.95)),
        max_scale_underflows=config.get("freeze_max_scale_underflows", 0),
    )


@dataclass(frozen=True)
class QATSchedule:
    soft_tokens: int
    hard_tokens: int
    freeze_eval_every_tokens: int
    initial_phase: str = "soft"
    temperature_start: float = 1.0
    temperature_end: float = 0.05
    freeze_indices_at_tokens: int | None = None
    freeze_max_tokens: int | None = None
    flip_threshold: float = 1e-4
    margin_threshold: float = 0.0
    sustain_evals: int = 3
    trend_tolerance: float = 0.01
    max_zero_fraction: float = 0.95
    max_scale_underflows: int = 0

    def __post_init__(self) -> None:
        if self.initial_phase not in {"soft", "hard", "frozen"}:
            raise ValueError("invalid initial QAT phase")
        if not all(math.isfinite(value) and value > 0 for value in (
                self.temperature_start, self.temperature_end)):
            raise ValueError("QAT temperatures must be finite and positive")
        token_values = {
            "soft_tokens": self.soft_tokens,
            "hard_tokens": self.hard_tokens,
            "freeze_eval_every_tokens": self.freeze_eval_every_tokens,
        }
        for name, value in token_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.freeze_eval_every_tokens == 0:
            raise ValueError("freeze_eval_every_tokens must be positive")
        if self.initial_phase == "soft" and self.soft_tokens == 0:
            raise ValueError("soft initial phase requires soft_tokens > 0")
        if self.initial_phase != "soft" and self.soft_tokens != 0:
            raise ValueError("non-soft initial phase requires soft_tokens = 0")
        if self.initial_phase == "frozen" and self.hard_tokens != 0:
            raise ValueError("frozen initial phase requires hard_tokens = 0")
        if isinstance(self.sustain_evals, bool) or not isinstance(self.sustain_evals, int) \
                or self.sustain_evals < 1:
            raise ValueError("sustain_evals must be a positive integer")
        if not all(math.isfinite(value) and value >= 0 for value in (
                self.flip_threshold, self.margin_threshold, self.trend_tolerance)):
            raise ValueError("QAT freeze thresholds must be finite and nonnegative")
        if not math.isfinite(self.max_zero_fraction) \
                or not 0 <= self.max_zero_fraction <= 1:
            raise ValueError("max_zero_fraction must be in [0,1]")
        if isinstance(self.max_scale_underflows, bool) \
                or not isinstance(self.max_scale_underflows, int) \
                or self.max_scale_underflows < 0:
            raise ValueError("max_scale_underflows must be a nonnegative integer")
        for name, value in (("freeze_indices_at_tokens", self.freeze_indices_at_tokens),
                            ("freeze_max_tokens", self.freeze_max_tokens)):
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        if self.freeze_max_tokens is not None \
                and self.freeze_max_tokens < self.earliest_freeze_tokens:
            raise ValueError("freeze_max_tokens precedes the earliest freeze point")

    @property
    def hard_start_tokens(self) -> int:
        return 0 if self.initial_phase in {"hard", "frozen"} else self.soft_tokens

    @property
    def earliest_freeze_tokens(self) -> int:
        configured = self.hard_start_tokens + self.hard_tokens
        return max(configured, self.freeze_indices_at_tokens or 0)

    @property
    def latest_freeze_tokens(self) -> int:
        return self.freeze_max_tokens if self.freeze_max_tokens is not None \
            else self.earliest_freeze_tokens + max(
                self.hard_tokens, self.freeze_eval_every_tokens)

    def validate_run(self, *, total_tokens: int, tokens_per_step: int) -> None:
        """Fail before model loading when a run cannot execute this schedule exactly."""
        for name, value in (("total_tokens", total_tokens),
                            ("tokens_per_step", tokens_per_step)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_phase == "frozen":
            return
        if total_tokens < self.latest_freeze_tokens:
            raise ValueError(
                f"total_tokens={total_tokens} cannot reach freeze_max_tokens="
                f"{self.latest_freeze_tokens}")
        boundaries = {
            "freeze_eval_every_tokens": self.freeze_eval_every_tokens,
            "hard_start_tokens": self.hard_start_tokens,
            "earliest_freeze_tokens": self.earliest_freeze_tokens,
            "latest_freeze_tokens": self.latest_freeze_tokens,
        }
        misaligned = {
            name: value for name, value in boundaries.items()
            if value and value % tokens_per_step
        }
        if misaligned:
            rendered = ", ".join(f"{name}={value}" for name, value in misaligned.items())
            raise ValueError(
                f"token schedule is not aligned to tokens_per_step={tokens_per_step}: "
                + rendered)
        if self.earliest_freeze_tokens % self.freeze_eval_every_tokens:
            raise ValueError("earliest freeze must fall on the freeze-evaluation cadence")
        if self.latest_freeze_tokens % self.freeze_eval_every_tokens:
            raise ValueError("latest freeze must fall on the freeze-evaluation cadence")
        hard_observations = (
            self.earliest_freeze_tokens // self.freeze_eval_every_tokens
            - self.hard_start_tokens // self.freeze_eval_every_tokens)
        if hard_observations < self.sustain_evals:
            raise ValueError(
                "hard phase cannot provide the required sustained freeze evaluations: "
                f"available={hard_observations}, required={self.sustain_evals}")


class QATController:
    def __init__(self, modules: Iterable[TQ1Linear], schedule: QATSchedule):
        self.modules = tuple(modules)
        if not self.modules:
            raise ValueError("QAT controller requires at least one TQ1Linear")
        self.schedule = schedule
        self.history: list[dict[str, float | int]] = []
        self.export_qualified = schedule.initial_phase == "frozen"
        self.failure_reason: str | None = None
        self.last_tokens = 0
        self.last_observation_tokens = 0
        for module in self.modules:
            if schedule.initial_phase != module.phase:
                module.set_phase(schedule.initial_phase)

    @property
    def phase(self) -> str:
        phases = {module.phase for module in self.modules}
        if len(phases) != 1:
            raise RuntimeError(f"TQ1 modules have divergent phases {sorted(phases)}")
        return next(iter(phases))

    def _temperature(self, tokens_seen: int) -> float:
        if self.schedule.soft_tokens <= 1:
            return self.schedule.temperature_end
        progress = min(max(tokens_seen, 0), self.schedule.soft_tokens) / \
            self.schedule.soft_tokens
        # Log-linear annealing is stable across large temperature ratios and is
        # fully determined by the global token position.
        return self.schedule.temperature_start * (
            self.schedule.temperature_end / self.schedule.temperature_start) ** progress

    def before_step(self, tokens_seen: int) -> None:
        if isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int) \
                or tokens_seen < 0:
            raise ValueError("tokens_seen must be a nonnegative integer")
        if tokens_seen != self.last_tokens:
            raise RuntimeError(
                f"QAT token position mismatch: controller={self.last_tokens}, "
                f"trainer={tokens_seen}")
        if self.failure_reason is not None:
            raise RuntimeError(self.failure_reason)
        if self.phase == "frozen":
            return
        if self.schedule.initial_phase == "soft" \
                and tokens_seen < self.schedule.soft_tokens:
            for module in self.modules:
                if module.phase != "soft":
                    module.set_phase("soft")
                module.set_temperature(self._temperature(tokens_seen))
            return
        for module in self.modules:
            if module.phase == "soft":
                module.set_phase("hard")

    def after_step(self, tokens_seen: int) -> None:
        if isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int) \
                or tokens_seen <= self.last_tokens:
            raise ValueError("post-step tokens_seen must advance monotonically")
        self.last_tokens = tokens_seen

    def observation_due(self, tokens_seen: int) -> bool:
        if tokens_seen != self.last_tokens:
            raise RuntimeError("freeze observation queried at a stale token position")
        return self.phase != "frozen" \
            and tokens_seen > self.last_observation_tokens \
            and tokens_seen % self.schedule.freeze_eval_every_tokens == 0

    def _recent_gates(self) -> tuple[bool, dict[str, bool]]:
        count = self.schedule.sustain_evals
        recent = self.history[-count:]
        enough = len(recent) == count
        flips = enough and all(0 <= item["flip_rate"] <= self.schedule.flip_threshold
                               for item in recent)
        margins = enough and all(math.isfinite(item["margin_p05"])
                                 and item["margin_p05"] >= self.schedule.margin_threshold
                                 for item in recent)
        trend = enough and all(
            math.isfinite(item[key]) for item in recent for key in ("val_ce", "kl_tf"))
        if trend:
            for key in ("val_ce", "kl_tf"):
                values = [item[key] for item in recent]
                if values[-1] > values[0] + self.schedule.trend_tolerance:
                    trend = False
        health = enough and all(
            0 <= item["zero_fraction_max"] <= self.schedule.max_zero_fraction
            and 0 <= item["scale_underflow_count"] <= self.schedule.max_scale_underflows
            for item in recent)
        gates = {
            "sustained": enough, "flip": flips, "margin": margins,
            "trend": trend, "health": health,
        }
        return all(gates.values()), gates

    def observe(self, tokens_seen: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        if not self.observation_due(tokens_seen):
            raise ValueError(
                f"tokens_seen={tokens_seen} is not a new freeze-evaluation boundary")
        record = {
            "tokens": int(tokens_seen),
            "flip_rate": float(metrics.get("flip_total", math.inf)),
            "margin_p05": float(metrics.get("tq1_margin_p05", -math.inf)),
            "val_ce": float(metrics.get("val_ce_primary", math.nan)),
            "kl_tf": float(metrics.get("kl_tf", math.nan)),
            "zero_fraction_max": float(
                metrics.get("tq1_zero_fraction_max", math.inf)),
            "scale_underflow_count": float(
                metrics.get("tq1_scale_underflow_count", math.inf)),
        }
        self.history.append(record)
        self.last_observation_tokens = tokens_seen
        eligible, gates = self._recent_gates()
        transitioned = False
        if self.failure_reason is not None:
            pass
        elif self.phase == "hard" \
                and tokens_seen >= self.schedule.earliest_freeze_tokens and eligible:
            for module in self.modules:
                module.freeze_indices()
            self.export_qualified = True
            transitioned = True
        elif self.phase != "frozen" \
                and tokens_seen >= self.schedule.latest_freeze_tokens:
            failed = [name for name, passed in gates.items() if not passed]
            self.failure_reason = "freeze gates unmet: " + ", ".join(failed)
        return self.status(transitioned=transitioned, gates=gates,
                           gate_observation=True)

    def status(self, *, transitioned: bool = False,
               gates: Mapping[str, bool] | None = None,
               gate_observation: bool = False) -> dict[str, Any]:
        if gates is None:
            _, gates = self._recent_gates()
        return {
            "phase": self.phase,
            "temperature": self.modules[0].temperature,
            "freeze_eligible": all(gates.values()),
            "freeze_gates": dict(gates),
            "gate_observation": gate_observation,
            "transitioned": transitioned,
            "export_qualified": self.export_qualified,
            "failure_reason": self.failure_reason,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 3,
            "domain": "global_tokens",
            "schedule": asdict(self.schedule),
            "history": [dict(item) for item in self.history],
            "phase": self.phase,
            "export_qualified": self.export_qualified,
            "failure_reason": self.failure_reason,
            "last_tokens": self.last_tokens,
            "last_observation_tokens": self.last_observation_tokens,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") == 1:
            raise ValueError(
                "legacy step-domain QAT checkpoint is not resumable under the "
                "token-domain schedule")
        if state.get("schema") == 2:
            raise ValueError(
                "controller schema 2 lacks fail-closed health-gate observations")
        if state.get("schema") != 3 or state.get("domain") != "global_tokens":
            raise ValueError("unsupported QAT controller checkpoint schema")
        if QATSchedule(**state["schedule"]) != self.schedule:
            raise ValueError("QAT schedule differs from the checkpoint")
        phase = str(state["phase"])
        for module in self.modules:
            if module.phase != phase:
                module.set_phase(phase)
        raw_history = state.get("history", [])
        history_keys = {
            "tokens", "flip_rate", "margin_p05", "val_ce", "kl_tf",
            "zero_fraction_max", "scale_underflow_count",
        }
        if not isinstance(raw_history, list):
            raise ValueError("checkpoint QAT history must be a list")
        self.history = []
        for item in raw_history:
            if not isinstance(item, Mapping) or set(item) != history_keys \
                    or isinstance(item["tokens"], bool) \
                    or not isinstance(item["tokens"], int) \
                    or any(isinstance(item[key], bool)
                           or not isinstance(item[key], (int, float))
                           for key in history_keys - {"tokens"}):
                raise ValueError("checkpoint has an invalid QAT history record")
            self.history.append({
                "tokens": item["tokens"],
                **{key: float(item[key]) for key in history_keys - {"tokens"}},
            })
        if not isinstance(state.get("export_qualified"), bool):
            raise ValueError("checkpoint export qualification must be boolean")
        self.export_qualified = state["export_qualified"]
        self.failure_reason = state.get("failure_reason")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise ValueError("checkpoint failure reason must be a string or null")
        positions = (state.get("last_tokens"), state.get("last_observation_tokens"))
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in positions):
            raise ValueError("checkpoint QAT token positions must be integers")
        self.last_tokens, self.last_observation_tokens = positions
        if self.last_tokens < 0 or self.last_observation_tokens < 0 \
                or self.last_observation_tokens > self.last_tokens:
            raise ValueError("checkpoint has invalid QAT token positions")
        if self.last_observation_tokens \
                and self.last_observation_tokens % self.schedule.freeze_eval_every_tokens:
            raise ValueError("checkpoint freeze observation is off the token cadence")
        history_tokens = [int(item.get("tokens", -1)) for item in self.history]
        if history_tokens != sorted(set(history_tokens)) \
                or (history_tokens and history_tokens[-1] != self.last_observation_tokens):
            raise ValueError("checkpoint has inconsistent QAT observation history")
        if any(tokens <= 0 or tokens % self.schedule.freeze_eval_every_tokens
               for tokens in history_tokens):
            raise ValueError("checkpoint QAT history is off the token cadence")
        expected_observation = (self.last_tokens // self.schedule.freeze_eval_every_tokens
                                * self.schedule.freeze_eval_every_tokens)
        if phase != "frozen" and self.last_observation_tokens != expected_observation:
            raise ValueError("checkpoint skipped a required freeze-gate observation")
        if self.export_qualified != (phase == "frozen"):
            raise ValueError("checkpoint phase and export qualification disagree")
        if self.export_qualified and self.failure_reason is not None:
            raise ValueError("checkpoint is both export-qualified and failed")

    def validate_position(self, tokens_seen: int) -> None:
        if tokens_seen != self.last_tokens:
            raise RuntimeError(
                f"checkpoint token mismatch: trainer={tokens_seen}, "
                f"controller={self.last_tokens}")

    def assert_export_qualified(self) -> None:
        if self.phase != "frozen" or not self.export_qualified:
            raise RuntimeError(self.failure_reason or "QAT indices are not frozen/export-qualified")
