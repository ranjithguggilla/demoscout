"""Scoring one release: a range, a hit probability, comparable tracks and plain reasons."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from .data import AUDIO, CATEGORICAL, FEATURES, HIT_THRESHOLD, NUMERIC

LABELS = {
    "danceability": "danceability",
    "energy": "energy",
    "key": "musical key",
    "loudness": "loudness",
    "mode": "mode",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "valence": "valence",
    "tempo": "tempo",
    "duration_min": "length in minutes",
    "release_year": "release year",
    "playlist_genre": "genre",
}


@dataclass
class Scout:
    """Everything the app needs, bundled so it can be saved and reloaded."""

    range_model: object
    hit_model: object
    reference: pd.Series
    catalogue: pd.DataFrame
    scaler: StandardScaler
    tree: KDTree = field(init=False)
    threshold: int = HIT_THRESHOLD

    def __post_init__(self):
        self.tree = KDTree(self.scaler.transform(self.catalogue[AUDIO]))

    def score(self, row: dict, n_similar: int = 5, n_reasons: int = 3) -> dict:
        return score_release(self, row, n_similar, n_reasons)


def make_reference(train: pd.DataFrame) -> pd.Series:
    ref = {c: float(train[c].median()) for c in NUMERIC}
    for c in CATEGORICAL:
        ref[c] = train[c].mode().iloc[0]
    return pd.Series(ref)[FEATURES]


def _frame(row: dict, reference: pd.Series) -> pd.DataFrame:
    merged = reference.to_dict()
    merged.update({k: v for k, v in row.items() if k in merged})
    return pd.DataFrame([merged])[FEATURES]


def reasons(scout: Scout, row: dict, top: int = 3) -> list[str]:
    """Swap one feature at a time to the typical value and report how far the median moves."""
    base_frame = _frame(row, scout.reference)
    base = float(scout.range_model.quantiles(base_frame, [0.5])[0, 0])
    deltas = []
    for col in FEATURES:
        if col not in row:
            continue
        swapped = base_frame.copy()
        swapped[col] = scout.reference[col]
        moved = base - float(scout.range_model.quantiles(swapped, [0.5])[0, 0])
        deltas.append((col, moved))
    deltas.sort(key=lambda t: -abs(t[1]))
    out = []
    for col, moved in deltas[:top]:
        if abs(moved) < 0.05:
            continue
        value = base_frame[col].iloc[0]
        ref = scout.reference[col]
        direction = "adds" if moved > 0 else "takes off"
        if col in CATEGORICAL:
            out.append(f"{LABELS[col]} '{value}' {direction} about {abs(moved):.1f} points against a typical release")
        else:
            out.append(f"{LABELS[col]} of {value:.2f} (typical {ref:.2f}) {direction} about {abs(moved):.1f} points")
    return out


def similar_tracks(scout: Scout, row: dict, n: int = 5) -> pd.DataFrame:
    vec = _frame(row, scout.reference)[AUDIO]
    _, idx = scout.tree.query(scout.scaler.transform(vec), k=n)
    cols = ["track_name", "track_artist", "playlist_genre", "track_popularity"]
    return scout.catalogue.iloc[idx[0]][cols].reset_index(drop=True)


def score_release(scout: Scout, row: dict, n_similar: int = 5, n_reasons: int = 3) -> dict:
    frame = _frame(row, scout.reference)
    lo, mid, hi = scout.range_model.quantiles(frame, [0.1, 0.5, 0.9])[0]
    p_hit = float(scout.hit_model.proba(frame)[0])
    return {
        "expected_popularity": float(mid),
        "low_80": float(lo),
        "high_80": float(hi),
        "hit_probability": p_hit,
        "reasons": reasons(scout, row, n_reasons),
        "similar": similar_tracks(scout, row, n_similar),
        "threshold": scout.threshold,
    }


def verdict(p_hit: float, base_rate: float, shortlist_cut: float) -> str:
    if p_hit >= shortlist_cut:
        return "shortlist"
    if p_hit >= base_rate:
        return "watch"
    return "pass"
