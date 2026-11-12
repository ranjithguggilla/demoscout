import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from demoscout import app, data, models, neighbors, scoring


@pytest.fixture(scope="module")
def scout(synthetic, split):
    tr = synthetic[split == "train"]
    rng = models.QuantileTrio().fit(tr[data.FEATURES], tr[data.TARGET])
    hit = models.hit_candidates()["logistic"].fit(tr[data.FEATURES], data.hit_label(tr[data.TARGET], 60))
    cols = ["track_name", "track_artist", "playlist_genre", "track_popularity", *data.AUDIO]
    scaler = StandardScaler().fit(tr[data.AUDIO])
    return scoring.Scout(rng, hit, scoring.make_reference(tr), tr[cols].reset_index(drop=True), scaler)


def test_screen_returns_complete_answer(scout):
    out = app.screen(scout, "pop", danceability=0.9, release_year=2019.9)
    assert out["low_80"] <= out["expected_popularity"] <= out["high_80"]
    assert 0 <= out["hit_probability"] <= 1
    assert out["verdict"] in {"shortlist", "watch", "pass"}
    assert len(out["similar"]) == 5
    text, table = app.describe(out)
    assert "Verdict" in text and isinstance(table, pd.DataFrame)


def test_newer_and_danceable_scores_higher(scout):
    new = app.screen(scout, "pop", danceability=0.9, release_year=2019.8)
    old = app.screen(scout, "pop", danceability=0.3, release_year=2009.0)
    assert new["expected_popularity"] > old["expected_popularity"]
    assert new["hit_probability"] > old["hit_probability"]


def test_reasons_name_the_driver(scout):
    reasons = scout.score({"release_year": 2009.0, "danceability": 0.25, "playlist_genre": "pop"})["reasons"]
    assert reasons and any("release year" in r or "danceability" in r for r in reasons)
    assert all("points" in r for r in reasons)


def test_unseen_genre_does_not_crash(scout):
    out = app.screen(scout, "polka")
    assert np.isfinite(out["expected_popularity"])


def test_verdict_thresholds():
    assert scoring.verdict(0.5, 0.15, 0.3) == "shortlist"
    assert scoring.verdict(0.2, 0.15, 0.3) == "watch"
    assert scoring.verdict(0.05, 0.15, 0.3) == "pass"


def test_similar_tracks_are_nearest_in_audio_space(scout):
    row = scout.catalogue.iloc[0][data.AUDIO].to_dict()
    near = scoring.similar_tracks(scout, {**row, "playlist_genre": "pop"}, n=3)
    assert near.iloc[0]["track_name"] == scout.catalogue.iloc[0]["track_name"]


def test_neighbor_benchmark_small(synthetic):
    A, scaler = neighbors.audio_matrix(synthetic)
    table = neighbors.benchmark(A[:600], A[600:700], k=5)
    assert {"brute force", "kd-tree"} <= set(table["index"])
    assert table.loc[table["index"] == "brute force", "recall_at_10"].iloc[0] == 1.0
    assert (table["recall_at_10"] > 0.5).all()
    assert neighbors.recall_at_k(np.array([[1, 2]]), np.array([[2, 3]])) == 0.5
    pred = neighbors.knn_popularity(A[:600], synthetic["track_popularity"][:600], A[600:620], k=5)
    assert pred.shape == (20,)
