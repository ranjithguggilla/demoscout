"""Charts for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, ACCENT, GREY, WARM = "#1f2937", "#0f766e", "#9ca3af", "#b45309"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def protocols(table, path: Path):
    """Skill is the share of mean-only squared error removed, so it compares across splits."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [WARM, GREY, ACCENT, INK]
    ax.bar(table["protocol"], table["skill"], color=colors[: len(table)])
    for i, v in enumerate(table["skill"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("skill over predicting the mean")
    ax.set_ylim(0, table["skill"].max() * 1.2)
    ax.set_title("Same model, four ways of cutting the data")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(levels, curves: dict, single: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0.4, 1], [0.4, 1], color=GREY, lw=1, ls="--", label="ideal")
    palette = [ACCENT, WARM, INK]
    for (name, vals), c in zip(curves.items(), palette, strict=False):
        ax.plot(levels, vals, marker="o", color=c, label=name)
    for (name, (lvl, val)), c in zip(single.items(), [INK], strict=False):
        ax.scatter([lvl], [val], marker="D", color=c, s=50, label=name, zorder=5)
    ax.set_xlabel("nominal interval level")
    ax.set_ylabel("share of test tracks inside")
    ax.set_title("Prediction intervals on unseen artists")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ann_tradeoff(table, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in table.iterrows():
        ax.scatter(r["recall_at_10"], r["queries_per_s"], s=60, color=ACCENT)
        ax.annotate(
            r["index"], (r["recall_at_10"], r["queries_per_s"]), fontsize=8, xytext=(5, 4), textcoords="offset points"
        )
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 against exact search")
    ax.set_ylabel("queries per second (log)")
    ax.set_title("Finding similar tracks: exact trees against approximate indexes")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def value_curve(shares, curves: dict, break_even_note: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    palette = {"model": ACCENT, "genre rule": WARM, "random": GREY, "hindsight": INK}
    for name, vals in curves.items():
        ax.plot(
            np.array(shares) * 100, vals, label=name, color=palette.get(name, INK), lw=2 if name == "model" else 1.3
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("share of releases promoted (%)")
    ax.set_ylabel("net value per 1,000 releases ($)")
    ax.set_title("Value of the shortlist " + break_even_note)
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hit_calibration(p, hit, path: Path, bins: int = 10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    xs = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    ys = [hit[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1)
    ax.plot(xs, ys, marker="o", color=ACCENT)
    ax.set_xlabel("predicted hit probability")
    ax.set_ylabel("observed hit rate")
    ax.set_title("Hit probability, ten equal-size bins")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
