import numpy as np

from demoscout import audit, data, models


def test_feature_blocks_cover_expected_sets(synthetic, split):
    out = audit.feature_blocks(synthetic[split == "train"], synthetic[split == "test"])
    assert "audio only" in set(out["feature_set"])
    full = out.set_index("feature_set").loc["all features", "rmse"]
    audio = out.set_index("feature_set").loc["audio only", "rmse"]
    assert full < audio  # release year and genre carry the planted signal


def test_permutation_importance_ranks_release_year_first(synthetic, split):
    tr, te = synthetic[split == "train"], synthetic[split == "test"]
    m = models.QuantileTrio().fit(tr[data.FEATURES], tr[data.TARGET])
    imp = audit.permutation_importance(m, te, repeats=2)
    assert imp.iloc[0]["feature"] == "release_year"


def test_artist_encoding_study_flags_overlap(synthetic):
    out = audit.artist_encoding_study(synthetic)
    row = out.set_index("split")
    assert row.loc["artist-disjoint", "test_rows_with_known_artist"] == 0.0
    assert row.loc["random rows", "test_rows_with_known_artist"] > 0.5


def test_group_audit_shapes(synthetic, split):
    te = synthetic[split == "test"]
    q = np.column_stack([te[data.TARGET] - 10, te[data.TARGET], te[data.TARGET] + 10]).clip(0, 100)
    out = audit.group_audit(te, q, np.full(len(te), 0.2))
    assert set(out["group"]) == {"genre", "era"}
    assert (out["coverage_80"] == 1.0).all()
