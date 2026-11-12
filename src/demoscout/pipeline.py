"""End-to-end run: select on validation, score once on test, audit, price, save."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import audit, data, economics, metrics, neighbors, plots
from .models import GaussianHit, NGBModel, candidates, hit_candidates
from .scoring import Scout, make_reference

LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]


def one_se_pick(order: list[str], losses: dict[str, np.ndarray]) -> tuple[str, dict]:
    """Simplest model whose mean loss is within one standard error of the best."""
    means = {k: float(v.mean()) for k, v in losses.items()}
    best = min(means, key=means.get)
    se = float(losses[best].std(ddof=1) / np.sqrt(len(losses[best])))
    for name in order:
        if name in means and means[name] <= means[best] + se:
            return name, {"best": best, "best_loss": means[best], "se": se, "pick_loss": means[name]}
    return best, {"best": best, "best_loss": means[best], "se": se, "pick_loss": means[best]}


def _round(x, nd=4):
    return None if x is None else round(float(x), nd)


def run(
    out_dir: Path = Path("outputs"),
    asset_dir: Path = Path("assets"),
    data_path: Path = data.DEFAULT_PATH,
    seed: int = 0,
    n_boot: int = 300,
    ngb_estimators: int = 400,
) -> dict:
    t0 = time.time()
    out_dir, asset_dir = Path(out_dir), Path(asset_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    raw = data.load_raw(data_path)
    df = data.clean(raw)
    split = data.artist_split(df, seed=seed)
    tr, va, te = df[split == "train"], df[split == "val"], df[split == "test"]
    Xtr, Xva, Xte = tr[data.FEATURES], va[data.FEATURES], te[data.FEATURES]
    ytr, yva, yte = (a[data.TARGET].to_numpy(dtype=float) for a in (tr, va, te))
    hit_tr, hit_va, hit_te = (data.hit_label(y) for y in (ytr, yva, yte))
    result: dict = {
        "data": {
            "raw_rows": int(len(raw)),
            "tracks": int(len(df)),
            "artists": int(df["track_artist"].nunique()),
            "train": int(len(tr)),
            "val": int(len(va)),
            "test": int(len(te)),
            "hit_threshold": data.HIT_THRESHOLD,
            "hit_rate": {"train": _round(hit_tr.mean()), "val": _round(hit_va.mean()), "test": _round(hit_te.mean())},
            "zero_popularity_share": _round((df[data.TARGET] == 0).mean()),
        }
    }

    # 1. range models: fit on train, choose on validation
    cands = candidates(seed, ngb_estimators)
    fitted, val_loss = {}, {}
    for name, m in cands.items():
        fitted[name] = m.fit(Xtr, ytr)
        val_loss[name] = metrics.pinball(yva, m.quantiles(Xva, metrics.QS))
    pick, info = one_se_pick(list(cands), val_loss)
    result["range_selection"] = {
        "metric": "mean pinball loss at q10/q50/q90, validation",
        "losses": {k: _round(v.mean()) for k, v in val_loss.items()},
        **{k: _round(v) if isinstance(v, float) else v for k, v in info.items()},
        "shipped": pick,
    }

    # 2. test scores for every candidate
    rows, test_q = [], {}
    for name, m in fitted.items():
        q = m.quantiles(Xte, metrics.QS)
        test_q[name] = q
        mu = m.mean(Xte)
        row = {
            "model": name,
            "rmse": float(np.sqrt(np.mean((mu - yte) ** 2))),
            "mae": float(np.mean(np.abs(mu - yte))),
            "pinball": float(metrics.pinball(yte, q).mean()),
            "coverage_80": metrics.coverage(yte, q[:, 0], q[:, 2]),
            "width_80": float(np.mean(q[:, 2] - q[:, 0])),
            "interval_score": float(metrics.interval_score(yte, q[:, 0], q[:, 2]).mean()),
            "auc_rank_by_mean": float(roc_auc_score(hit_te, mu)),
        }
        if hasattr(m, "params"):
            mu_, sd_ = m.params(Xte)
            row["crps"] = float(metrics.crps_gaussian(yte, mu_, sd_).mean())
        rows.append(row)
    range_table = pd.DataFrame(rows)
    range_table.to_csv(out_dir / "tables" / "range_models_test.csv", index=False)
    result["range_test"] = {r["model"]: {k: _round(v) for k, v in r.items() if k != "model"} for r in rows}

    # paired cluster bootstrap on the shipped model
    groups = te["track_artist"].to_numpy()
    shipped_loss = metrics.pinball(yte, test_q[pick])
    cis = {}
    for ref in ("mean_only", "ridge"):
        gap = metrics.pinball(yte, test_q[ref]) - shipped_loss
        p, lo, hi = metrics.cluster_bootstrap(groups, lambda i, g=gap: float(g[i].mean()), n_boot, seed)
        cis[f"pinball_gain_vs_{ref}"] = [_round(p), _round(lo), _round(hi)]
    mu_s = fitted[pick].mean(Xte)
    p, lo, hi = metrics.cluster_bootstrap(
        groups, lambda i: float(np.sqrt(np.mean((mu_s[i] - yte[i]) ** 2))), n_boot, seed
    )
    cis["rmse"] = [_round(p), _round(lo), _round(hi)]
    result["range_ci"] = cis

    # 3. hit models
    hit_fit = {}
    for name, h in hit_candidates().items():
        hit_fit[name] = h.fit(Xtr, hit_tr)
    ngb = fitted["ngboost_normal"]
    hit_fit["ngboost_tail"] = GaussianHit(ngb, data.HIT_THRESHOLD, "ngboost_tail")
    hit_order = ["base_rate", "genre_logit", "logistic", "hgb_classifier", "ngboost_tail"]
    ll_val = {
        n: -(
            hit_va * np.log(np.clip(h.proba(Xva), 1e-6, 1)) + (1 - hit_va) * np.log(np.clip(1 - h.proba(Xva), 1e-6, 1))
        )
        for n, h in hit_fit.items()
    }
    hit_pick, hinfo = one_se_pick(hit_order, ll_val)
    hit_rows = []
    for n, h in hit_fit.items():
        p = h.proba(Xte)
        hit_rows.append({"model": n, **metrics.classification_block(hit_te, p)})
    hit_table = pd.DataFrame(hit_rows)
    hit_table.to_csv(out_dir / "tables" / "hit_models_test.csv", index=False)
    result["hit_selection"] = {
        "metric": "log loss, validation",
        "losses": {k: _round(v.mean()) for k, v in ll_val.items()},
        "shipped": hit_pick,
        "se": _round(hinfo["se"]),
    }
    result["hit_test"] = {r["model"]: {k: _round(v) for k, v in r.items() if k != "model"} for r in hit_rows}
    p_hit_te = hit_fit[hit_pick].proba(Xte)
    p_hit_va = hit_fit[hit_pick].proba(Xva)
    auc_ci = metrics.cluster_bootstrap(groups, lambda i: float(roc_auc_score(hit_te[i], p_hit_te[i])), n_boot, seed)
    result["hit_auc_ci"] = [_round(x) for x in auc_ci]

    # 4. interval reliability
    curves = {}
    for name in ("hgb_gaussian", "ngboost_normal", "ridge"):
        m = fitted[name]
        vals = []
        for lvl in LEVELS:
            q = m.quantiles(Xte, [(1 - lvl) / 2, (1 + lvl) / 2])
            vals.append(metrics.coverage(yte, q[:, 0], q[:, 1]))
        curves[name] = vals
    result["interval_reliability"] = {"levels": LEVELS, **{k: [_round(x) for x in v] for k, v in curves.items()}}
    plots.reliability(
        LEVELS,
        curves,
        {"hgb_quantile (q10-q90)": (0.8, result["range_test"]["hgb_quantile"]["coverage_80"])},
        asset_dir / "interval_reliability.png",
    )
    plots.hit_calibration(p_hit_te, hit_te, asset_dir / "hit_calibration.png")

    # 5. audits
    protocols = audit.split_protocols(raw, seed)
    protocols.to_csv(out_dir / "tables" / "split_protocols.csv", index=False)
    plots.protocols(protocols, asset_dir / "split_protocols.png")
    enc = audit.artist_encoding_study(df, seed)
    blocks = audit.feature_blocks(tr, te)
    perm = audit.permutation_importance(fitted[pick], te, repeats=5, seed=seed)
    groups_tbl = audit.group_audit(te, test_q[pick], p_hit_te)
    for name, t in [
        ("artist_encoding", enc),
        ("feature_blocks", blocks),
        ("permutation_importance", perm),
        ("group_audit", groups_tbl),
    ]:
        t.to_csv(out_dir / "tables" / f"{name}.csv", index=False)
    result["audit"] = {
        "protocols": protocols.round(4).to_dict("records"),
        "artist_encoding": enc.round(4).to_dict("records"),
        "feature_blocks": blocks.round(4).to_dict("records"),
        "permutation_top": perm.head(6).round(4).to_dict("records"),
        "groups": groups_tbl.round(4).to_dict("records"),
    }

    # 6. similar-track search
    A_tr, scaler = neighbors.audio_matrix(tr)
    A_te, _ = neighbors.audio_matrix(te, scaler)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(len(A_te), size=min(3000, len(A_te)), replace=False)
    ann = neighbors.benchmark(A_tr, A_te[qidx], k=10, seed=seed)
    ann.to_csv(out_dir / "tables" / "neighbor_search.csv", index=False)
    plots.ann_tradeoff(ann, asset_dir / "neighbor_search.png")
    knn_pred = neighbors.knn_popularity(A_tr, ytr, A_te, k=20)
    result["neighbors"] = {
        "table": ann.round(4).to_dict("records"),
        "knn20_rmse": _round(np.sqrt(np.mean((knn_pred - yte) ** 2))),
        "n_queries": int(len(qidx)),
        "train_points": int(len(A_tr)),
        "dims": int(A_tr.shape[1]),
    }

    # 7. economics
    shares = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    genre_rate = tr.groupby("playlist_genre")["track_popularity"].apply(lambda s: (s >= data.HIT_THRESHOLD).mean())
    rule_te = te["playlist_genre"].map(genre_rate).to_numpy() + rng.random(len(te)) * 1e-6
    rand_te = rng.random(len(te))
    best = economics.best_share(hit_va, p_hit_va, shares)
    result["economics"] = {
        "assumptions": {
            "cost_per_promoted": economics.COST,
            "value_per_hit": economics.VALUE_HIT,
            "break_even_precision": economics.break_even_precision(),
        },
        "share_chosen_on_validation": best,
    }
    curves_v = {
        "model": [economics.net_per_thousand(hit_te, p_hit_te, s) for s in shares],
        "genre rule": [economics.net_per_thousand(hit_te, rule_te, s) for s in shares],
        "random": [economics.net_per_thousand(hit_te, rand_te, s) for s in shares],
        "hindsight": [economics.net_per_thousand(hit_te, hit_te + rng.random(len(te)) * 1e-6, s) for s in shares],
    }
    plots.value_curve(shares, curves_v, r"(cost \$2,500, hit worth \$10,000)", asset_dir / "value_curve.png")
    pol = {}
    for name, sc in [("model", p_hit_te), ("genre_rule", rule_te), ("random", rand_te)]:
        stat = lambda i, sc=sc: economics.net_per_thousand(hit_te[i], sc[i], best)  # noqa: E731
        p, lo, hi = metrics.cluster_bootstrap(groups, stat, n_boot, seed)
        pol[name] = {
            "net_per_1000": _round(p, 0),
            "ci": [_round(lo, 0), _round(hi, 0)],
            "precision": _round(economics.precision_at(hit_te, sc, best)),
        }

    def gap(i):
        return economics.net_per_thousand(hit_te[i], p_hit_te[i], best) - economics.net_per_thousand(
            hit_te[i], rule_te[i], best
        )

    p, lo, hi = metrics.cluster_bootstrap(groups, gap, n_boot, seed)
    pol["model_minus_genre_rule"] = [_round(p, 0), _round(lo, 0), _round(hi, 0)]
    result["economics"]["policies"] = pol
    result["economics"]["hindsight_net_per_1000"] = _round(
        economics.net_per_thousand(hit_te, hit_te + 1e-6 * rng.random(len(te)), best), 0
    )
    grid = []
    for value in (5000, 7500, 10000, 15000):
        for cost in (1500, 2500, 4000):
            grid.append(
                {
                    "value_per_hit": value,
                    "cost": cost,
                    "model": _round(economics.net_per_thousand(hit_te, p_hit_te, best, cost, value), 0),
                    "genre_rule": _round(economics.net_per_thousand(hit_te, rule_te, best, cost, value), 0),
                }
            )
    pd.DataFrame(grid).to_csv(out_dir / "tables" / "economics_sensitivity.csv", index=False)
    result["economics"]["sensitivity"] = grid
    pd.DataFrame({"share": shares, **curves_v}).to_csv(out_dir / "tables" / "value_curve.csv", index=False)

    # 8. deployment bundle: refit the winners on train plus validation
    trva = pd.concat([tr, va])
    range_final = candidates(seed, ngb_estimators)[pick].fit(trva[data.FEATURES], trva[data.TARGET])
    hit_final = {**hit_candidates()}.get(hit_pick) or GaussianHit(
        NGBModel(seed=seed).fit(trva[data.FEATURES], trva[data.TARGET]), data.HIT_THRESHOLD
    )
    if hasattr(hit_final, "fit") and not isinstance(hit_final, GaussianHit):
        hit_final.fit(trva[data.FEATURES], data.hit_label(trva[data.TARGET]))
    cols = ["track_name", "track_artist", "playlist_genre", "track_popularity", *data.AUDIO]
    scaler_final = neighbors.audio_matrix(trva)[1]
    scout = Scout(range_final, hit_final, make_reference(trva), trva[cols].reset_index(drop=True), scaler_final)
    joblib.dump(scout, out_dir / "scout.joblib")

    result["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
