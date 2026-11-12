"""Promotion economics: what a ranked shortlist is worth under stated assumptions."""

from __future__ import annotations

import numpy as np

# Assumptions, all adjustable. A promoted release costs COST. A promoted release that
# turns out to be a hit earns VALUE_HIT of incremental net revenue. Neither number is
# observed in the data; the sensitivity tables show how the answer moves.
COST = 2500.0
VALUE_HIT = 10000.0


def shortlist(scores: np.ndarray, share: float) -> np.ndarray:
    n = max(1, int(round(len(scores) * share)))
    order = np.argsort(-scores, kind="stable")
    return order[:n]


def net_per_thousand(
    hit: np.ndarray, scores: np.ndarray, share: float, cost: float = COST, value: float = VALUE_HIT
) -> float:
    pick = shortlist(scores, share)
    n_pick = len(pick)
    gain = value * hit[pick].sum() - cost * n_pick
    return float(gain / len(scores) * 1000.0)


def precision_at(hit: np.ndarray, scores: np.ndarray, share: float) -> float:
    return float(hit[shortlist(scores, share)].mean())


def break_even_precision(cost: float = COST, value: float = VALUE_HIT) -> float:
    return cost / value


def best_share(hit: np.ndarray, scores: np.ndarray, grid, cost: float = COST, value: float = VALUE_HIT) -> float:
    vals = [net_per_thousand(hit, scores, s, cost, value) for s in grid]
    return float(grid[int(np.argmax(vals))])
