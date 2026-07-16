"""Deterministic facility-location construction for TQ1 J and P codebooks."""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .codebook import (
    Codebook,
    base3_ids,
    product_codebook,
    sign_canonical_codebook,
)
from .ptq import ternary_universe


@dataclass(frozen=True)
class PatternCorpus:
    demand: torch.Tensor
    diagonal: torch.Tensor | None = None
    covariance: torch.Tensor | None = None

    def __post_init__(self) -> None:
        demand = self.demand.detach().double().cpu().contiguous()
        if tuple(demand.shape) != (6561,) or torch.any(demand < 0) \
                or not torch.isfinite(demand).all() or float(demand.sum()) <= 0:
            raise ValueError("pattern demand must be finite nonnegative [6561] with mass")
        object.__setattr__(self, "demand", demand)
        if self.diagonal is not None and self.covariance is not None:
            raise ValueError("pattern corpus uses diagonal or covariance, not both")
        if self.diagonal is not None:
            diagonal = self.diagonal.detach().double().cpu().contiguous()
            if tuple(diagonal.shape) != (6561, 8) or torch.any(diagonal < 0) \
                    or not torch.isfinite(diagonal).all():
                raise ValueError("pattern diagonal must be finite nonnegative [6561,8]")
            object.__setattr__(self, "diagonal", diagonal)
        if self.covariance is not None:
            covariance = self.covariance.detach().double().cpu().contiguous()
            if tuple(covariance.shape) != (6561, 8, 8) or not torch.isfinite(covariance).all():
                raise ValueError("pattern covariance must be finite [6561,8,8]")
            if not torch.allclose(
                    covariance, covariance.transpose(-1, -2),
                    atol=1e-10, rtol=1e-10):
                raise ValueError("pattern covariance must be symmetric")
            minimum = float(torch.linalg.eigvalsh(covariance).min())
            if minimum < -1e-9:
                raise ValueError(
                    f"pattern covariance must be positive semidefinite (min={minimum})")
            object.__setattr__(self, "covariance", covariance)

    @classmethod
    def from_counts(cls, counts: torch.Tensor) -> "PatternCorpus":
        return cls(torch.as_tensor(counts, dtype=torch.float64))


def combine_pattern_corpora(corpora: Sequence[PatternCorpus], *,
                            normalize_each: bool = False) -> PatternCorpus:
    """Combine demand and conditional sensitivity without averaging averages."""
    items = list(corpora)
    if not items:
        raise ValueError("cannot combine an empty pattern-corpus inventory")
    diagonal = items[0].diagonal is not None
    covariance = items[0].covariance is not None
    if any((item.diagonal is not None) != diagonal
           or (item.covariance is not None) != covariance for item in items):
        raise ValueError("pattern corpora use incompatible sensitivity metrics")
    demand = torch.zeros(6561, dtype=torch.float64)
    metric_sum = None
    if diagonal:
        metric_sum = torch.zeros((6561, 8), dtype=torch.float64)
    elif covariance:
        metric_sum = torch.zeros((6561, 8, 8), dtype=torch.float64)
    for item in items:
        scale = 1.0 / float(item.demand.sum()) if normalize_each else 1.0
        weighted_demand = item.demand * scale
        demand += weighted_demand
        metric = item.diagonal if diagonal else item.covariance
        if metric is not None and metric_sum is not None:
            multiplier = weighted_demand.reshape(
                -1, *([1] * (metric.ndim - 1)))
            metric_sum += metric * multiplier
    if metric_sum is None:
        return PatternCorpus(demand)
    averaged = torch.zeros_like(metric_sum)
    positive = demand > 0
    divisor = demand[positive].reshape(-1, *([1] * (metric_sum.ndim - 1)))
    averaged[positive] = metric_sum[positive] / divisor
    return (PatternCorpus(demand, diagonal=averaged) if diagonal
            else PatternCorpus(demand, covariance=averaged))


def _family(name: str) -> str:
    for family in ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"):
        if name.endswith(family) or f".{family}." in name:
            return family
    return name


def corpus_from_tensors(patterns: Mapping[str, torch.Tensor], *,
                        weighting: str = "family_equal") -> PatternCorpus:
    """Aggregate already-ternary groups under an explicit corpus weighting."""
    if weighting not in {"parameter", "tensor_equal", "family_equal"}:
        raise ValueError("unsupported corpus weighting")
    encoded: dict[str, torch.Tensor] = {}
    for name, value in patterns.items():
        raw = value.detach().cpu()
        if raw.numel() % 8:
            raise ValueError(f"{name}: pattern corpus size is not divisible by eight")
        try:
            encoded[name] = base3_ids(raw.reshape(-1, 8))
        except ValueError as exc:
            raise ValueError(f"{name}: pattern corpus contains a non-trit") from exc
    if not encoded:
        raise ValueError("pattern corpus is empty")
    counts = torch.zeros(6561, dtype=torch.float64)
    if weighting == "parameter":
        for values in encoded.values():
            counts += torch.bincount(values, minlength=6561).double()
    elif weighting == "tensor_equal":
        for values in encoded.values():
            counts += torch.bincount(values, minlength=6561).double() / values.numel()
    else:
        families: dict[str, list[torch.Tensor]] = {}
        for name, values in encoded.items():
            families.setdefault(_family(name), []).append(values)
        for values_by_tensor in families.values():
            family_groups = sum(value.numel() for value in values_by_tensor)
            for values in values_by_tensor:
                counts += torch.bincount(values, minlength=6561).double() / family_groups
    return PatternCorpus(counts)


def sensitivity_corpus_from_tensors(
        patterns: Mapping[str, torch.Tensor], *,
        diagonal: Mapping[str, torch.Tensor] | None = None,
        covariance: Mapping[str, torch.Tensor] | None = None,
        weighting: str = "family_equal", row_chunk: int = 64) -> PatternCorpus:
    """Aggregate `[N,K]` trits and group sensitivities by base-3 pattern.

    Calibration sensitivity is shared by every output row of one matrix.  Row
    chunking avoids materializing an `[N,K/8,8,8]` covariance expansion for
    large FFN projections while retaining exact FP64 weighted sums.
    """
    if weighting not in {"parameter", "tensor_equal", "family_equal"}:
        raise ValueError("unsupported corpus weighting")
    if (diagonal is None) == (covariance is None):
        raise ValueError("sensitivity corpus requires exactly one metric inventory")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk < 1:
        raise ValueError("sensitivity corpus row chunk must be positive")
    metric_inventory = diagonal if diagonal is not None else covariance
    assert metric_inventory is not None
    if not patterns or set(patterns) != set(metric_inventory):
        raise ValueError("pattern and sensitivity tensor inventories must match exactly")

    encoded: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    group_counts: dict[str, int] = {}
    for name, value in patterns.items():
        trits = value.detach().cpu()
        if trits.ndim != 2 or trits.shape[1] % 8:
            raise ValueError(f"{name}: sensitivity patterns must have shape [N,K], K%8=0")
        rows, width = trits.shape
        try:
            encoded[name] = base3_ids(trits.reshape(rows, width // 8, 8))
        except ValueError as exc:
            raise ValueError(f"{name}: pattern corpus contains a non-trit") from exc
        metric = metric_inventory[name].detach().double().cpu().contiguous()
        expected = ((width // 8, 8) if diagonal is not None
                    else (width // 8, 8, 8))
        if diagonal is not None and tuple(metric.shape) == (width,):
            metric = metric.reshape(expected)
        if tuple(metric.shape) != expected or not torch.isfinite(metric).all():
            raise ValueError(f"{name}: sensitivity shape must be {expected} and finite")
        if diagonal is not None and torch.any(metric < 0):
            raise ValueError(f"{name}: diagonal sensitivity must be nonnegative")
        metrics[name] = metric
        group_counts[name] = rows * (width // 8)

    coefficients: dict[str, float] = {}
    if weighting == "parameter":
        coefficients = {name: 1.0 for name in patterns}
    elif weighting == "tensor_equal":
        coefficients = {name: 1.0 / group_counts[name] for name in patterns}
    else:
        family_counts: dict[str, int] = {}
        for name, count in group_counts.items():
            family_counts[_family(name)] = family_counts.get(_family(name), 0) + count
        coefficients = {
            name: 1.0 / family_counts[_family(name)] for name in patterns
        }

    demand = torch.zeros(6561, dtype=torch.float64)
    metric_sum = torch.zeros(
        (6561, 8) if diagonal is not None else (6561, 8, 8),
        dtype=torch.float64)
    for name, ids in encoded.items():
        rows, groups = ids.shape
        metric = metrics[name]
        coefficient = coefficients[name]
        for start in range(0, rows, row_chunk):
            count = min(row_chunk, rows - start)
            flat_ids = ids[start:start + count].reshape(-1)
            weights = torch.full(
                (flat_ids.numel(),), coefficient, dtype=torch.float64)
            demand.index_add_(0, flat_ids, weights)
            expanded = metric.unsqueeze(0).expand(count, *metric.shape) \
                .reshape(flat_ids.numel(), *metric.shape[1:])
            metric_sum.index_add_(
                0, flat_ids, expanded * coefficient)
    positive = demand > 0
    averaged = torch.zeros_like(metric_sum)
    divisor = demand[positive].reshape(-1, *([1] * (metric_sum.ndim - 1)))
    averaged[positive] = metric_sum[positive] / divisor
    return (PatternCorpus(demand, diagonal=averaged) if diagonal is not None
            else PatternCorpus(demand, covariance=averaged))


def canonical_shapes() -> torch.Tensor:
    universe = ternary_universe()
    nonzero = universe != 0
    first = nonzero.long().argmax(1)
    negative = nonzero.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    return universe[torch.unique(base3_ids(canonical), sorted=True)]


def required_anchor_rows(shapes: torch.Tensor) -> list[int]:
    support = (shapes != 0).sum(1)
    anchors = torch.nonzero((support == 0) | (support <= 2) | (support == 8)).flatten()
    zero = int(torch.nonzero(support == 0).flatten()[0])
    return [zero] + [int(index) for index in anchors if int(index) != zero]


def _distance_matrix(corpus: PatternCorpus, candidates: torch.Tensor, *,
                     lambda_nz: float = 0.0, chunk: int = 256) -> torch.Tensor:
    if not math.isfinite(lambda_nz) or lambda_nz < 0:
        raise ValueError("lambda_nz must be finite and nonnegative")
    patterns = ternary_universe().double()
    result = torch.empty((patterns.shape[0], candidates.shape[0]), dtype=torch.float64)
    for start in range(0, candidates.shape[0], chunk):
        selected = candidates[start:start + chunk].double()
        plus = patterns[:, None] - selected[None]
        minus = patterns[:, None] + selected[None]
        if corpus.covariance is not None:
            d_plus = torch.einsum("uci,uij,ucj->uc", plus, corpus.covariance, plus)
            d_minus = torch.einsum("uci,uij,ucj->uc", minus, corpus.covariance, minus)
        else:
            diagonal = (corpus.diagonal if corpus.diagonal is not None
                        else torch.ones((6561, 8), dtype=torch.float64))
            d_plus = (plus.square() * diagonal[:, None]).sum(-1)
            d_minus = (minus.square() * diagonal[:, None]).sum(-1)
        penalty = lambda_nz * (selected != 0).sum(1).double()[None]
        result[:, start:start + selected.shape[0]] = torch.minimum(d_plus, d_minus) + penalty
    return result


def facility_location_select(corpus: PatternCorpus, *, select_count: int,
                             lambda_nz: float = 0.0, swap_limit: int = 0,
                             return_trace: bool = False) \
        -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact greedy additions followed by deterministic best-improvement swaps."""
    shapes = canonical_shapes()
    anchors = required_anchor_rows(shapes)
    if select_count < len(anchors) or select_count > shapes.shape[0]:
        raise ValueError(f"select_count must be in [{len(anchors)}, {shapes.shape[0]}]")
    distances = _distance_matrix(corpus, shapes, lambda_nz=lambda_nz)
    demand = corpus.demand
    selected = list(anchors)
    selected_set = set(selected)
    min_distance = distances[:, selected].min(1).values
    objectives = [float((demand * min_distance).sum())]
    # Facility-location gains are monotone non-increasing as the selected set
    # grows.  A lazy-greedy heap therefore returns the exact same next row as a
    # full rescan, while recomputing only bounds that can still win.  Heap ties
    # use canonical/base-3 row order, matching torch.argmax's first maximum.
    gains = torch.empty(shapes.shape[0], dtype=torch.float64)
    for start in range(0, shapes.shape[0], 256):
        candidate = distances[:, start:start + 256]
        improvement = (min_distance[:, None]
                       - torch.minimum(min_distance[:, None], candidate)).clamp_min(0)
        gains[start:start + candidate.shape[1]] = (demand[:, None] * improvement).sum(0)
    gains[torch.tensor(selected)] = -torch.inf
    heap = [(-float(gains[row]), row) for row in range(shapes.shape[0])
            if row not in selected_set]
    heapq.heapify(heap)
    lazy_recomputations = 0
    while len(selected) < select_count:
        if not heap:
            raise RuntimeError("facility-location heap became empty")
        _, incoming = heapq.heappop(heap)
        candidate_distance = distances[:, incoming]
        improvement = (min_distance
                       - torch.minimum(min_distance, candidate_distance)).clamp_min(0)
        exact_gain = float((demand * improvement).sum())
        lazy_recomputations += 1
        if heap:
            next_bound, next_row = -heap[0][0], heap[0][1]
            safe = exact_gain > next_bound or \
                (exact_gain == next_bound and incoming < next_row)
        else:
            safe = True
        if not safe:
            heapq.heappush(heap, (-exact_gain, incoming))
            continue
        selected.append(incoming)
        selected_set.add(incoming)
        min_distance = torch.minimum(min_distance, candidate_distance)
        objectives.append(float((demand * min_distance).sum()))

    anchor_set = set(anchors)
    termination = "swap_limit" if swap_limit == 0 else "no_improvement"
    swaps = []
    for pass_index in range(swap_limit):
        selected_tensor = torch.tensor(selected)
        selected_dist = distances[:, selected_tensor]
        order = torch.argsort(selected_dist, dim=1, stable=True)
        nearest_pos = order[:, 0]
        nearest = selected_dist.gather(1, nearest_pos[:, None]).squeeze(1)
        second = selected_dist.gather(1, order[:, 1:2]).squeeze(1)
        current = float((demand * nearest).sum())
        # Objective for adding each candidate without a removal. Removing one
        # selected row changes only the demand points assigned to that row.
        # Those clusters partition the corpus, reducing a swap pass from
        # O(|S|*|U|*|C|) to O(|U|*|C|), exactly and deterministically.
        common = torch.empty(shapes.shape[0], dtype=torch.float64)
        for start in range(0, shapes.shape[0], 256):
            rows = slice(start, min(start + 256, shapes.shape[0]))
            common[rows] = (demand[:, None] * torch.minimum(
                nearest[:, None], distances[:, rows])).sum(0)
        common[torch.tensor(selected)] = torch.inf
        best = None
        for outgoing_pos, outgoing in enumerate(selected):
            if outgoing in anchor_set:
                continue
            assigned = nearest_pos == outgoing_pos
            objective = common.clone()
            if torch.any(assigned):
                assigned_demand = demand[assigned]
                old = nearest[assigned]
                replacement = second[assigned]
                for start in range(0, shapes.shape[0], 256):
                    rows = slice(start, min(start + 256, shapes.shape[0]))
                    candidate_distance = distances[assigned, rows]
                    correction = assigned_demand[:, None] * (
                        torch.minimum(replacement[:, None], candidate_distance)
                        - torch.minimum(old[:, None], candidate_distance))
                    objective[rows] += correction.sum(0)
            incoming = int(torch.argmin(objective))
            candidate = (float(objective[incoming]), incoming, outgoing, outgoing_pos)
            if best is None or candidate < best:
                best = candidate
        if best is None or best[0] >= current:
            termination = "no_improvement"
            break
        _, incoming, outgoing, outgoing_pos = best
        selected[outgoing_pos] = incoming
        selected_set.remove(outgoing)
        selected_set.add(incoming)
        objective = float((demand * distances[:, torch.tensor(selected)].min(1).values).sum())
        swaps.append({"pass": pass_index, "outgoing_base3": int(base3_ids(shapes[outgoing])),
                      "incoming_base3": int(base3_ids(shapes[incoming])),
                      "objective": objective})
        objectives.append(objective)
    selected_shapes = shapes[torch.tensor(selected)]
    report = {
        "algorithm": "exact_greedy_facility_location+best_improvement_swaps",
        "select_count": select_count,
        "anchor_count": len(anchors),
        "anchor_base3_ids": [
            int(value) for value in base3_ids(shapes[torch.tensor(anchors)]).tolist()
        ],
        "lambda_nz": lambda_nz,
        "objectives": objectives if return_trace else [objectives[0], objectives[-1]],
        "swaps": swaps,
        "termination": termination,
        "swap_limit": swap_limit,
        "lazy_gain_recomputations": lazy_recomputations,
    }
    return selected_shapes, report


def build_joint_codebook(codebook_id: str, index_format: str, corpus: PatternCorpus, *,
                         scope: str = "model", lambda_nz: float = 0.0,
                         swap_limit: int = 4) -> Codebook:
    count = 1024 if index_format == "v11" else 2048
    shapes, report = facility_location_select(
        corpus, select_count=count, lambda_nz=lambda_nz, swap_limit=swap_limit)
    return sign_canonical_codebook(codebook_id, index_format, shapes, scope=scope,
                                   provenance=report)


def _half_universe() -> torch.Tensor:
    value = torch.arange(81, dtype=torch.int64)
    lanes = []
    for _ in range(4):
        lanes.append((value % 3 - 1).to(torch.int8)); value //= 3
    return torch.stack(lanes, 1)


def _canonical_halves() -> torch.Tensor:
    universe = _half_universe()
    nonzero = universe != 0
    first = nonzero.long().argmax(1)
    negative = nonzero.any(1) & (universe.gather(1, first[:, None]).squeeze(1) < 0)
    canonical = universe * torch.where(negative, -1, 1).to(torch.int8)[:, None]
    return universe[torch.unique(base3_ids(canonical), sorted=True)]


def _marginal_counts(corpus: PatternCorpus, first_half: bool) -> torch.Tensor:
    universe = ternary_universe()
    ids = base3_ids(universe[:, :4] if first_half else universe[:, 4:])
    counts = torch.zeros(81, dtype=torch.float64)
    counts.scatter_add_(0, ids, corpus.demand)
    return counts


def _select_sign_free_half(counts: torch.Tensor, count: int = 32) -> torch.Tensor:
    candidates = _canonical_halves()
    universe = _half_universe().double()
    distance = torch.minimum(
        (universe[:, None] - candidates.double()[None]).square().sum(-1),
        (universe[:, None] + candidates.double()[None]).square().sum(-1),
    )
    zero = int(torch.nonzero((candidates == 0).all(1)).flatten()[0])
    selected = [zero]
    minimum = distance[:, zero]
    while len(selected) < count:
        gains = (counts[:, None] * (minimum[:, None]
                 - torch.minimum(minimum[:, None], distance)).clamp_min(0)).sum(0)
        gains[torch.tensor(selected)] = -torch.inf
        incoming = int(torch.argmax(gains))
        selected.append(incoming)
        minimum = torch.minimum(minimum, distance[:, incoming])
    rows = candidates[torch.tensor(selected)]
    zero_row = rows[(rows == 0).all(1)]
    nonzero = rows[~(rows == 0).all(1)]
    return torch.cat((zero_row, nonzero[torch.argsort(base3_ids(nonzero))]))


def _product_objective(corpus: PatternCorpus, a: torch.Tensor, b: torch.Tensor,
                       *, chunk: int = 256) -> float:
    base = torch.cat((a[:, None, :].expand(-1, b.shape[0], -1),
                      b[None, :, :].expand(a.shape[0], -1, -1)), dim=-1).reshape(-1, 8)
    patterns = ternary_universe().double()
    minimum = torch.full((6561,), torch.inf, dtype=torch.float64)
    for start in range(0, base.shape[0], chunk):
        code = base[start:start + chunk].double()
        plus = patterns[:, None] - code[None]
        minus = patterns[:, None] + code[None]
        if corpus.covariance is not None:
            d1 = torch.einsum("uci,uij,ucj->uc", plus, corpus.covariance, plus)
            d2 = torch.einsum("uci,uij,ucj->uc", minus, corpus.covariance, minus)
        else:
            diagonal = (corpus.diagonal if corpus.diagonal is not None
                        else torch.ones((6561, 8), dtype=torch.float64))
            d1 = (plus.square() * diagonal[:, None]).sum(-1)
            d2 = (minus.square() * diagonal[:, None]).sum(-1)
        minimum = torch.minimum(minimum, torch.minimum(d1, d2).min(1).values)
    return float((corpus.demand * minimum).sum())


def build_product_codebook(codebook_id: str, index_format: str, corpus: PatternCorpus, *,
                           scope: str = "model", swap_limit: int = 2) -> Codebook:
    a = _select_sign_free_half(_marginal_counts(corpus, True))
    if index_format == "v11":
        b = _select_sign_free_half(_marginal_counts(corpus, False))
    elif index_format == "v12":
        canonical = _canonical_halves()
        # Structural requirement: all 40 sign-pair representatives plus zero.
        b = canonical
        opposites = -canonical[1:]
        selected_opposites: list[int] = []
        for _ in range(23):
            best = None
            for row in range(opposites.shape[0]):
                if row in selected_opposites:
                    continue
                candidate = torch.cat((b, opposites[torch.tensor(selected_opposites + [row])]))
                objective = _product_objective(corpus, a, candidate)
                key = (objective, int(base3_ids(opposites[row:row + 1])), row)
                if best is None or key < best:
                    best = key
            assert best is not None
            selected_opposites.append(best[2])
        b = torch.cat((b, opposites[torch.tensor(selected_opposites)]))
        zero = b[(b == 0).all(1)]
        nonzero = b[~(b == 0).all(1)]
        b = torch.cat((zero, nonzero[torch.argsort(base3_ids(nonzero))]))
    else:
        raise ValueError("product index format must be v11 or v12")
    objectives = [_product_objective(corpus, a, b)]
    termination = "swap_limit"
    # Deterministic best-improvement row swaps, A then B, preserving structure.
    for _ in range(swap_limit):
        improved = False
        for table_name in ("a", "b"):
            table = a if table_name == "a" else b
            if index_format == "v12" and table_name == "b":
                continue  # B's exact sign-pair multiplicity is structural, not optional.
            universe = _canonical_halves()
            existing = set(base3_ids(table).tolist())
            current = objectives[-1]
            best = None
            for outgoing in range(1, table.shape[0]):
                for incoming in range(1, universe.shape[0]):
                    incoming_id = int(base3_ids(universe[incoming:incoming + 1]))
                    if incoming_id in existing:
                        continue
                    candidate = table.clone(); candidate[outgoing] = universe[incoming]
                    zero = candidate[(candidate == 0).all(1)]
                    nonzero = candidate[~(candidate == 0).all(1)]
                    candidate = torch.cat((zero, nonzero[torch.argsort(base3_ids(nonzero))]))
                    objective = _product_objective(
                        corpus, candidate if table_name == "a" else a,
                        candidate if table_name == "b" else b)
                    key = (objective, incoming_id, int(base3_ids(table[outgoing:outgoing + 1])))
                    if best is None or key < best[0]:
                        best = (key, candidate)
            if best is not None and best[0][0] < current:
                if table_name == "a": a = best[1]
                else: b = best[1]
                objectives.append(best[0][0]); improved = True
        if not improved:
            termination = "no_improvement"
            break
    return product_codebook(codebook_id, index_format, a, b, scope=scope,
                            provenance={
                                "algorithm": "marginal_medoid+structured_full_objective_swaps",
                                "initialization_objective": objectives[0],
                                "objectives": objectives,
                                "swap_limit": swap_limit,
                                "termination": termination,
                            })
