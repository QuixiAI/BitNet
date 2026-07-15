"""Resumable soft/hard/frozen TQ1 QAT curriculum and export gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .qat import TQ1Linear


@dataclass(frozen=True)
class QATSchedule:
    initial_phase: str = "soft"
    temperature_start: float = 1.0
    temperature_end: float = 0.05
    soft_steps: int = 1000
    hard_steps: int = 4000
    freeze_indices_at: int | None = None
    freeze_max_step: int | None = None
    flip_threshold: float = 1e-4
    margin_threshold: float = 0.0
    sustain_evals: int = 3
    trend_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.initial_phase not in {"soft", "hard", "frozen"}:
            raise ValueError("invalid initial QAT phase")
        if not all(math.isfinite(value) and value > 0 for value in (
                self.temperature_start, self.temperature_end)):
            raise ValueError("QAT temperatures must be finite and positive")
        if self.soft_steps < 0 or self.hard_steps < 0 or self.sustain_evals < 1:
            raise ValueError("QAT step counts must be nonnegative and sustain_evals positive")
        if self.flip_threshold < 0 or self.margin_threshold < 0 \
                or self.trend_tolerance < 0:
            raise ValueError("QAT freeze thresholds must be nonnegative")
        if self.freeze_indices_at is not None and self.freeze_indices_at < 0:
            raise ValueError("freeze_indices_at must be nonnegative")
        if self.freeze_max_step is not None and self.freeze_max_step < 0:
            raise ValueError("freeze_max_step must be nonnegative")

    @property
    def hard_start(self) -> int:
        return 0 if self.initial_phase == "hard" else self.soft_steps

    @property
    def earliest_freeze(self) -> int:
        configured = self.hard_start + self.hard_steps
        return max(configured, self.freeze_indices_at or 0)

    @property
    def latest_freeze(self) -> int:
        return self.freeze_max_step if self.freeze_max_step is not None \
            else self.earliest_freeze + max(self.hard_steps, 1)


class QATController:
    def __init__(self, modules: Iterable[TQ1Linear], schedule: QATSchedule):
        self.modules = tuple(modules)
        if not self.modules:
            raise ValueError("QAT controller requires at least one TQ1Linear")
        self.schedule = schedule
        self.history: list[dict[str, float]] = []
        self.export_qualified = schedule.initial_phase == "frozen"
        self.failure_reason: str | None = None
        self.last_step = 0
        for module in self.modules:
            if schedule.initial_phase != module.phase:
                module.set_phase(schedule.initial_phase)

    @property
    def phase(self) -> str:
        phases = {module.phase for module in self.modules}
        if len(phases) != 1:
            raise RuntimeError(f"TQ1 modules have divergent phases {sorted(phases)}")
        return next(iter(phases))

    def _temperature(self, step: int) -> float:
        if self.schedule.soft_steps <= 1:
            return self.schedule.temperature_end
        progress = min(max(step, 0), self.schedule.soft_steps - 1) / (
            self.schedule.soft_steps - 1)
        # Log-linear annealing is stable across large temperature ratios and is
        # fully determined by integer global step.
        return self.schedule.temperature_start * (
            self.schedule.temperature_end / self.schedule.temperature_start) ** progress

    def before_step(self, step: int) -> None:
        self.last_step = int(step)
        if self.phase == "frozen":
            return
        if self.schedule.initial_phase == "soft" and step < self.schedule.soft_steps:
            for module in self.modules:
                if module.phase != "soft":
                    module.set_phase("soft")
                module.set_temperature(self._temperature(step))
            return
        for module in self.modules:
            if module.phase == "soft":
                module.set_phase("hard")

    def _recent_gates(self) -> tuple[bool, dict[str, bool]]:
        count = self.schedule.sustain_evals
        recent = self.history[-count:]
        enough = len(recent) == count
        flips = enough and all(item["flip_rate"] <= self.schedule.flip_threshold
                               for item in recent)
        margins = enough and all(item["margin_p05"] >= self.schedule.margin_threshold
                                 for item in recent)
        trend = enough
        if enough:
            for key in ("val_ce", "kl_tf"):
                values = [item[key] for item in recent if math.isfinite(item[key])]
                if len(values) >= 2 and values[-1] > values[0] + self.schedule.trend_tolerance:
                    trend = False
        gates = {"sustained": enough, "flip": flips, "margin": margins, "trend": trend}
        return all(gates.values()), gates

    def observe(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "step": float(step),
            "flip_rate": float(metrics.get("flip_total", math.inf)),
            "margin_p05": float(metrics.get("tq1_margin_p05", -math.inf)),
            "val_ce": float(metrics.get("val_ce_primary", math.nan)),
            "kl_tf": float(metrics.get("kl_tf", math.nan)),
        }
        self.history.append(record)
        eligible, gates = self._recent_gates()
        transitioned = False
        if self.phase == "hard" and step >= self.schedule.earliest_freeze and eligible:
            for module in self.modules:
                module.freeze_indices()
            self.export_qualified = True
            transitioned = True
        elif self.phase != "frozen" and step >= self.schedule.latest_freeze:
            failed = [name for name, passed in gates.items() if not passed]
            self.failure_reason = "freeze gates unmet: " + ", ".join(failed)
        return {
            "phase": self.phase,
            "temperature": self.modules[0].temperature,
            "freeze_eligible": eligible,
            "freeze_gates": gates,
            "transitioned": transitioned,
            "export_qualified": self.export_qualified,
            "failure_reason": self.failure_reason,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "schedule": asdict(self.schedule),
            "history": self.history,
            "phase": self.phase,
            "export_qualified": self.export_qualified,
            "failure_reason": self.failure_reason,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != 1:
            raise ValueError("unsupported QAT controller checkpoint schema")
        if QATSchedule(**state["schedule"]) != self.schedule:
            raise ValueError("QAT schedule differs from the checkpoint")
        phase = str(state["phase"])
        for module in self.modules:
            if module.phase != phase:
                module.set_phase(phase)
        self.history = [dict(item) for item in state.get("history", [])]
        self.export_qualified = bool(state.get("export_qualified", False))
        self.failure_reason = state.get("failure_reason")
        self.last_step = int(state.get("last_step", 0))

    def assert_export_qualified(self) -> None:
        if self.phase != "frozen" or not self.export_qualified:
            raise RuntimeError(self.failure_reason or "QAT indices are not frozen/export-qualified")
