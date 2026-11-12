"""Candidate models. All of them expose the same small interface.

``fit(X, y)``, ``mean(X)``, ``quantiles(X, qs)`` and, for the parametric
ones, ``params(X)`` returning a mean and a standard deviation per row.
Preprocessing sits inside each estimator, so nothing is learned from
validation or test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import CATEGORICAL, NUMERIC

LOW, HIGH = 0.0, 100.0


def make_preprocessor(numeric: list[str] | None = None, categorical: list[str] | None = None):
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    parts = []
    if numeric:
        parts.append(("num", StandardScaler(), list(numeric)))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)))
    return ColumnTransformer(parts, remainder="drop")


class GaussianModel:
    """Point model plus a constant spread taken from out-of-fold residuals."""

    def __init__(self, name: str, regressor, numeric=None, categorical=None, seed: int = 0):
        self.name = name
        self.regressor = regressor
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed
        self.pipe_: Pipeline | None = None
        self.sigma_: float = float("nan")

    def _pipe(self) -> Pipeline:
        from sklearn.base import clone

        return Pipeline(
            [
                ("prep", make_preprocessor(self.numeric, self.categorical)),
                ("reg", clone(self.regressor)),
            ]
        )

    def fit(self, X: pd.DataFrame, y) -> GaussianModel:
        y = np.asarray(y, dtype=float)
        oof = cross_val_predict(self._pipe(), X, y, cv=KFold(5, shuffle=True, random_state=self.seed))
        self.sigma_ = float(np.std(y - oof, ddof=1))
        self.pipe_ = self._pipe().fit(X, y)
        return self

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict(X)

    def params(self, X: pd.DataFrame):
        mu = self.mean(X)
        return mu, np.full(len(mu), self.sigma_)

    def quantiles(self, X: pd.DataFrame, qs) -> np.ndarray:
        mu, sd = self.params(X)
        return np.clip(mu[:, None] + sd[:, None] * norm.ppf(np.asarray(qs))[None, :], LOW, HIGH)


class NGBModel:
    """Natural-gradient boosting with a Normal head: mean and spread both depend on the row."""

    def __init__(
        self,
        name: str = "ngboost_normal",
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        seed: int = 0,
    ):
        self.name = name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.prep_ = None
        self.ngb_ = None

    def fit(self, X: pd.DataFrame, y) -> NGBModel:
        self.prep_ = make_preprocessor().fit(X)
        self.ngb_ = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=40),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=0.8,
            random_state=self.seed,
            verbose=False,
        )
        self.ngb_.fit(self.prep_.transform(X), np.asarray(y, dtype=float))
        return self

    def params(self, X: pd.DataFrame):
        dist = self.ngb_.pred_dist(self.prep_.transform(X))
        return dist.params["loc"], dist.params["scale"]

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.params(X)[0]

    def quantiles(self, X: pd.DataFrame, qs) -> np.ndarray:
        mu, sd = self.params(X)
        return np.clip(mu[:, None] + sd[:, None] * norm.ppf(np.asarray(qs))[None, :], LOW, HIGH)


class QuantileTrio:
    """Three boosted models trained with pinball loss, one per quantile."""

    QS = (0.1, 0.5, 0.9)

    def __init__(self, name: str = "hgb_quantile", seed: int = 0):
        self.name = name
        self.seed = seed
        self.prep_ = None
        self.models_: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, X: pd.DataFrame, y) -> QuantileTrio:
        self.prep_ = make_preprocessor().fit(X)
        Z = self.prep_.transform(X)
        for q in self.QS:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=self.seed,
            )
            self.models_[q] = m.fit(Z, np.asarray(y, dtype=float))
        return self

    def quantiles(self, X: pd.DataFrame, qs) -> np.ndarray:
        Z = self.prep_.transform(X)
        known = np.column_stack([self.models_[q].predict(Z) for q in self.QS])
        known = np.sort(known, axis=1)
        out = np.column_stack([np.interp(q, self.QS, known[i]) for i in range(len(known)) for q in qs])
        return np.clip(out.reshape(len(known), len(qs)), LOW, HIGH)

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.quantiles(X, [0.5])[:, 0]


def candidates(seed: int = 0, ngb_estimators: int = 400) -> dict[str, object]:
    """Ordered simplest to most complex; the order feeds the one-standard-error rule."""
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )
    return {
        "mean_only": GaussianModel("mean_only", DummyRegressor(), [], [], seed),
        "genre_mean": GaussianModel("genre_mean", Ridge(alpha=1.0), [], CATEGORICAL, seed),
        "ridge": GaussianModel("ridge", Ridge(alpha=10.0), seed=seed),
        "hgb_gaussian": GaussianModel("hgb_gaussian", hgb, seed=seed),
        "hgb_quantile": QuantileTrio(seed=seed),
        "ngboost_normal": NGBModel(n_estimators=ngb_estimators, seed=seed),
    }


class HitClassifier:
    """Probability that a release reaches the hit threshold."""

    def __init__(self, name: str, estimator, numeric=None, categorical=None):
        self.name = name
        self.estimator = estimator
        self.numeric = numeric
        self.categorical = categorical
        self.pipe_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y_hit) -> HitClassifier:
        from sklearn.base import clone

        self.pipe_ = Pipeline(
            [
                ("prep", make_preprocessor(self.numeric, self.categorical)),
                ("clf", clone(self.estimator)),
            ]
        ).fit(X, np.asarray(y_hit))
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(X)[:, 1]


class GaussianHit:
    """Hit probability read off a fitted Normal forecast (mean and spread)."""

    def __init__(self, model, threshold: float, name: str | None = None):
        self.model = model
        self.threshold = threshold
        self.name = name or f"{model.name}_tail"

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        mu, sd = self.model.params(X)
        return 1.0 - norm.cdf((self.threshold - 0.5 - mu) / np.maximum(sd, 1e-6))


def hit_candidates() -> dict[str, HitClassifier]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "base_rate": HitClassifier("base_rate", DummyClassifier(strategy="prior"), [], []),
        "genre_logit": HitClassifier("genre_logit", LogisticRegression(max_iter=500), [], CATEGORICAL),
        "logistic": HitClassifier("logistic", LogisticRegression(C=0.5, max_iter=2000)),
        "hgb_classifier": HitClassifier(
            "hgb_classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ),
        ),
    }
