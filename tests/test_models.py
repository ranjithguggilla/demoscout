import numpy as np

from demoscout import data, models
from demoscout.pipeline import one_se_pick


def _fit_all(synthetic, split):
    tr = synthetic[split == "train"]
    out = {}
    for name, m in models.candidates().items():
        if name == "ngboost_normal":
            m = models.NGBModel(n_estimators=30)
        out[name] = m.fit(tr[data.FEATURES], tr[data.TARGET])
    return out, tr


def test_every_candidate_returns_ordered_quantiles(synthetic, split):
    fitted, _ = _fit_all(synthetic, split)
    te = synthetic[split == "test"]
    for name, m in fitted.items():
        q = m.quantiles(te[data.FEATURES], [0.1, 0.5, 0.9])
        assert q.shape == (len(te), 3), name
        assert (q[:, 0] <= q[:, 1] + 1e-9).all() and (q[:, 1] <= q[:, 2] + 1e-9).all(), name
        assert q.min() >= 0 and q.max() <= 100


def test_gradient_models_find_the_planted_signal(synthetic, split):
    fitted, _ = _fit_all(synthetic, split)
    te = synthetic[split == "test"]
    y = te[data.TARGET].to_numpy()
    rmse = {n: np.sqrt(np.mean((m.mean(te[data.FEATURES]) - y) ** 2)) for n, m in fitted.items()}
    assert rmse["hgb_gaussian"] < rmse["mean_only"] * 0.8
    assert rmse["ridge"] < rmse["mean_only"]


def test_gaussian_spread_is_positive_and_constant(synthetic, split):
    fitted, _ = _fit_all(synthetic, split)
    te = synthetic[split == "test"]
    mu, sd = fitted["ridge"].params(te[data.FEATURES])
    assert (sd > 0).all() and np.allclose(sd, sd[0])
    assert mu.shape == sd.shape


def test_ngboost_spread_varies_by_row(synthetic, split):
    fitted, _ = _fit_all(synthetic, split)
    te = synthetic[split == "test"]
    _, sd = fitted["ngboost_normal"].params(te[data.FEATURES])
    assert sd.std() > 0


def test_preprocessing_does_not_see_test_rows(synthetic, split):
    tr = synthetic[split == "train"]
    te = synthetic[split == "test"].copy()
    m = models.candidates()["ridge"].fit(tr[data.FEATURES], tr[data.TARGET])
    before = m.mean(tr[data.FEATURES].head(20))
    te["danceability"] = 99.0
    m.mean(te[data.FEATURES])
    assert np.allclose(before, m.mean(tr[data.FEATURES].head(20)))


def test_hit_models_give_probabilities(synthetic, split):
    tr, te = synthetic[split == "train"], synthetic[split == "test"]
    hit = data.hit_label(tr[data.TARGET], 50)
    for name, h in models.hit_candidates().items():
        p = h.fit(tr[data.FEATURES], hit).proba(te[data.FEATURES])
        assert ((p >= 0) & (p <= 1)).all(), name


def test_one_se_pick_prefers_simpler_within_tolerance():
    rng = np.random.default_rng(0)
    base = rng.normal(5, 1, 400)
    losses = {"simple": base + 0.01, "complex": base}
    pick, info = one_se_pick(["simple", "complex"], losses)
    assert pick == "simple" and info["best"] == "complex"
    losses = {"simple": base + 1.0, "complex": base}
    assert one_se_pick(["simple", "complex"], losses)[0] == "complex"
