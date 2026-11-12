"""Checks on how much of the headline number survives honest conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from . import data, metrics
from .models import make_preprocessor


def _point_model(numeric=None, categorical=None, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical)),
            (
                "reg",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _score(train: pd.DataFrame, test: pd.DataFrame, numeric=None, categorical=None) -> dict:
    model = _point_model(numeric, categorical).fit(train[data.FEATURES], train[data.TARGET])
    pred = model.predict(test[data.FEATURES])
    y = test[data.TARGET].to_numpy(dtype=float)
    mse = float(np.mean((pred - y) ** 2))
    base = float(np.mean((train[data.TARGET].mean() - y) ** 2))
    return {"rmse": mse**0.5, "skill": 1 - mse / base, "n_test": len(test)}


def split_protocols(raw: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The same model under four ways of cutting the data."""
    with_repeats = data.clean(raw, dedupe=False)
    deduped = data.clean(raw, dedupe=True)
    rows = []
    for label, frame, splitter in [
        ("random rows, repeats kept", with_repeats, data.random_split),
        ("random rows, repeats removed", deduped, data.random_split),
        ("artist-disjoint", deduped, data.artist_split),
        ("forward in time", deduped, data.forward_split),
    ]:
        s = splitter(frame) if splitter is data.forward_split else splitter(frame, seed=seed)
        out = _score(frame[s == "train"], frame[s == "test"])
        rows.append({"protocol": label, **out})
    return pd.DataFrame(rows)


def artist_encoding_study(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add each artist's average popularity as a feature, with and without artist overlap.

    Training rows get a leave-one-out mean so the model cannot read its own label.
    """
    rows = []
    for label, splitter in [("random rows", data.random_split), ("artist-disjoint", data.artist_split)]:
        s = splitter(df, seed=seed)
        train, test = df[s == "train"].copy(), df[s == "test"].copy()
        grand = train[data.TARGET].mean()
        grp = train.groupby("track_artist")[data.TARGET].agg(["sum", "count"])
        t_sum = train["track_artist"].map(grp["sum"])
        t_cnt = train["track_artist"].map(grp["count"])
        train["artist_mean"] = ((t_sum - train[data.TARGET]) / (t_cnt - 1)).where(t_cnt > 1, grand)
        seen = test["track_artist"].map(grp["sum"]) / test["track_artist"].map(grp["count"])
        test["artist_mean"] = seen.fillna(grand)
        feats = [*data.FEATURES, "artist_mean"]
        pipe = Pipeline(
            [
                ("prep", make_preprocessor(data.NUMERIC + ["artist_mean"], data.CATEGORICAL)),
                ("reg", _point_model().named_steps["reg"]),
            ]
        ).fit(train[feats], train[data.TARGET])
        pred = pipe.predict(test[feats])
        y = test[data.TARGET].to_numpy(dtype=float)
        rows.append(
            {
                "split": label,
                "rmse_with_artist_mean": float(np.sqrt(np.mean((pred - y) ** 2))),
                "test_rows_with_known_artist": float(test["track_artist"].isin(grp.index).mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_blocks(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    spec = {
        "all features": (data.NUMERIC, data.CATEGORICAL),
        "no release year": ([c for c in data.NUMERIC if c != "release_year"], data.CATEGORICAL),
        "no genre": (data.NUMERIC, []),
        "audio only": (data.AUDIO, []),
        "release year and genre only": (["release_year"], data.CATEGORICAL),
    }
    rows = []
    for name, (num, cat) in spec.items():
        rows.append({"feature_set": name, **_score(train, test, num, cat)})
    return pd.DataFrame(rows)


def permutation_importance(model, test: pd.DataFrame, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = test[data.FEATURES]
    y = test[data.TARGET].to_numpy(dtype=float)
    base = metrics.pinball(y, model.quantiles(X, metrics.QS)).mean()
    rows = []
    for col in data.FEATURES:
        bumps = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            bumps.append(metrics.pinball(y, model.quantiles(shuffled, metrics.QS)).mean() - base)
        rows.append({"feature": col, "pinball_increase": float(np.mean(bumps)), "sd": float(np.std(bumps))})
    return pd.DataFrame(rows).sort_values("pinball_increase", ascending=False).reset_index(drop=True)


def group_audit(test: pd.DataFrame, q: np.ndarray, p_hit: np.ndarray) -> pd.DataFrame:
    """Error, interval coverage and hit rates by genre and by release era."""
    y = test[data.TARGET].to_numpy(dtype=float)
    hit = data.hit_label(y)
    era = pd.cut(
        test["release_year"],
        [0, 2015, 2018, 2019, 3000],
        labels=["before 2015", "2015-17", "2018", "2019 on"],
        right=False,
    )
    out = []
    for kind, keys in [("genre", test["playlist_genre"]), ("era", era)]:
        for key in pd.unique(keys.dropna()):
            m = (keys == key).to_numpy()
            if m.sum() < 30:
                continue
            out.append(
                {
                    "group": kind,
                    "value": str(key),
                    "n": int(m.sum()),
                    "rmse": float(np.sqrt(np.mean((q[m, 1] - y[m]) ** 2))),
                    "coverage_80": metrics.coverage(y[m], q[m, 0], q[m, 2]),
                    "mean_predicted_hit": float(p_hit[m].mean()),
                    "observed_hit": float(hit[m].mean()),
                }
            )
    return pd.DataFrame(out)
