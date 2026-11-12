"""Scoring rules and bootstrap helpers."""

from __future__ import annotations

import numpy as np
import properscoring as ps
from scipy.stats import norm
from sklearn.metrics import log_loss, roc_auc_score

QS = (0.1, 0.5, 0.9)


def pinball(y, q_pred, qs=QS) -> np.ndarray:
    """Per-row pinball loss averaged over the given quantile levels."""
    y = np.asarray(y, dtype=float)[:, None]
    q = np.asarray(qs)[None, :]
    diff = y - q_pred
    return np.maximum(q * diff, (q - 1) * diff).mean(axis=1)


def crps_gaussian(y, mu, sd) -> np.ndarray:
    return ps.crps_gaussian(np.asarray(y, dtype=float), mu=mu, sig=np.maximum(sd, 1e-6))


def interval_score(y, lo, hi, alpha: float = 0.2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def hit_probability(mu, sd, threshold: float) -> np.ndarray:
    return 1.0 - norm.cdf((threshold - 0.5 - mu) / np.maximum(sd, 1e-6))


def ece(y_true, p, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(total)


def classification_block(y_true, p) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
        "ece": ece(y_true, p),
    }


def cluster_bootstrap(groups, stat_fn, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """Resample whole artists, not rows. ``stat_fn`` receives an index array."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(len(groups)))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        draws.append(stat_fn(idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)
