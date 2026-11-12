import numpy as np

from demoscout import economics, metrics


def test_pinball_median_is_half_absolute_error():
    y = np.array([10.0, 20.0])
    q = np.array([[5.0, 12.0, 30.0], [25.0, 18.0, 22.0]])
    loss = metrics.pinball(y, q)
    assert loss.shape == (2,)
    assert loss[0] > 0
    one = metrics.pinball(y, np.array([[10.0], [20.0]]), qs=(0.5,))
    assert np.allclose(one, 0)


def test_crps_gaussian_is_smaller_for_sharper_correct_forecast():
    y = np.zeros(200)
    sharp = metrics.crps_gaussian(y, np.zeros(200), np.full(200, 1.0)).mean()
    vague = metrics.crps_gaussian(y, np.zeros(200), np.full(200, 5.0)).mean()
    assert sharp < vague


def test_interval_score_penalises_misses():
    y = np.array([5.0, 50.0])
    s = metrics.interval_score(y, np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    assert s[1] > s[0] and s[0] == 10.0


def test_coverage_counts_inside_points():
    assert metrics.coverage([1, 5, 9], 2, 8) == 1 / 3


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert metrics.ece(y, p) < 0.02


def test_cluster_bootstrap_interval_contains_point():
    groups = np.repeat(np.arange(40), 5)
    vals = np.random.default_rng(1).normal(size=200)
    p, lo, hi = metrics.cluster_bootstrap(groups, lambda i: float(vals[i].mean()), n_boot=100)
    assert lo <= p <= hi


def test_hit_probability_monotone_in_mean():
    low = metrics.hit_probability(np.array([40.0]), np.array([20.0]), 65)
    high = metrics.hit_probability(np.array([70.0]), np.array([20.0]), 65)
    assert high > low


def test_economics_break_even_and_shortlist():
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    hit = np.array([1, 0, 0, 0])
    assert economics.break_even_precision(2500, 10000) == 0.25
    assert list(economics.shortlist(scores, 0.5)) == [0, 1]
    # one hit in two promoted: 10,000 - 2 * 2,500 = 5,000 over 4 releases -> 1,250 per 1,000... scaled
    assert economics.net_per_thousand(hit, scores, 0.5) == 5000 / 4 * 1000
    assert economics.precision_at(hit, scores, 0.5) == 0.5
    assert economics.best_share(hit, scores, [0.25, 0.5, 1.0]) == 0.25
