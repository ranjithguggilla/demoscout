"""Nearest-neighbour search over audio features: exact trees against approximate indexes."""

from __future__ import annotations

import time

import hnswlib
import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import AUDIO


def audio_matrix(df: pd.DataFrame, scaler: StandardScaler | None = None):
    scaler = scaler or StandardScaler().fit(df[AUDIO])
    return scaler.transform(df[AUDIO]).astype("float32"), scaler


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth, strict=True))
    return hits / truth.size


def benchmark(train: np.ndarray, queries: np.ndarray, k: int = 10, seed: int = 0) -> pd.DataFrame:
    """Build and query each index; recall is measured against the exact brute-force answer."""
    rows = []

    brute, tb = _timed(lambda: NearestNeighbors(algorithm="brute").fit(train))
    truth, tq = _timed(lambda: brute.kneighbors(queries, n_neighbors=k, return_distance=False))
    rows.append(("brute force", tb, tq, 1.0))

    tree, tb = _timed(lambda: KDTree(train, leaf_size=40))
    found, tq = _timed(lambda: tree.query(queries, k=k, return_distance=False))
    rows.append(("kd-tree", tb, tq, recall_at_k(found, truth)))

    for ef in (16, 64, 256):

        def build():
            idx = hnswlib.Index(space="l2", dim=train.shape[1])
            idx.init_index(max_elements=len(train), ef_construction=200, M=16, random_seed=seed)
            idx.add_items(train, num_threads=1)
            return idx

        idx, tb = _timed(build)
        idx.set_ef(ef)
        (found, _), tq = _timed(lambda index=idx: index.knn_query(queries, k=k, num_threads=1))
        rows.append((f"hnsw (ef={ef})", tb, tq, recall_at_k(found, truth)))

    nnd, tb = _timed(lambda: NNDescent(train, n_neighbors=15, random_state=seed, n_jobs=1))
    nnd.prepare()
    (found, _), tq = _timed(lambda: nnd.query(queries, k=k, epsilon=0.1))
    rows.append(("nn-descent", tb, tq, recall_at_k(found, truth)))

    table = pd.DataFrame(rows, columns=["index", "build_s", "query_s", "recall_at_10"])
    table["queries_per_s"] = len(queries) / table["query_s"]
    return table


def knn_popularity(train: np.ndarray, y_train, queries: np.ndarray, k: int = 20) -> np.ndarray:
    """Mean popularity of the k closest training tracks, as a deliberately plain predictor."""
    tree = KDTree(train)
    idx = tree.query(queries, k=k, return_distance=False)
    return np.asarray(y_train, dtype=float)[idx].mean(axis=1)
