"""Gradio front end. The scoring logic lives in :func:`screen` so it can be tested without a browser."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import AUDIO
from .scoring import Scout, verdict

BASE_RATE = 0.15
SHORTLIST_CUT = 0.30

DEFAULTS = {
    "danceability": 0.65, "energy": 0.70, "key": 5, "loudness": -6.0, "mode": 1,
    "speechiness": 0.06, "acousticness": 0.15, "instrumentalness": 0.0, "liveness": 0.18,
    "valence": 0.50, "tempo": 120.0, "duration_min": 3.5, "release_year": 2019.5,
}  # fmt: skip


def screen(scout: Scout, genre: str, **audio) -> dict:
    """Score one release and add a plain verdict. Returns plain python types."""
    row = {**DEFAULTS, **audio, "playlist_genre": genre}
    out = scout.score(row)
    out["verdict"] = verdict(out["hit_probability"], BASE_RATE, SHORTLIST_CUT)
    return out


def describe(out: dict) -> tuple[str, pd.DataFrame]:
    lines = [
        f"**Verdict: {out['verdict']}**",
        "",
        f"Expected popularity {out['expected_popularity']:.0f} "
        f"(80% range {out['low_80']:.0f} to {out['high_80']:.0f}).",
        f"Chance of reaching {out['threshold']} or more: {out['hit_probability']:.0%}.",
        "",
        "What moves it:",
    ]
    lines += [f"- {r}" for r in out["reasons"]] or ["- nothing stands out from a typical release"]
    return "\n".join(lines), out["similar"]


def launch(bundle: Path = Path("outputs/scout.joblib")):
    import gradio as gr

    scout: Scout = joblib.load(bundle)
    genres = sorted(scout.catalogue["playlist_genre"].unique())

    def handler(genre, danceability, energy, loudness, speechiness, acousticness,
                instrumentalness, valence, tempo, duration_min, release_year):  # fmt: skip
        out = screen(
            scout, genre, danceability=danceability, energy=energy, loudness=loudness,
            speechiness=speechiness, acousticness=acousticness, instrumentalness=instrumentalness,
            valence=valence, tempo=tempo, duration_min=duration_min, release_year=release_year,
        )  # fmt: skip
        return describe(out)

    with gr.Blocks(title="demoscout") as demo:
        gr.Markdown("# demoscout\nScreen a new release from its audio profile.")
        genre = gr.Dropdown(genres, value="pop", label="genre")
        sliders = [gr.Slider(0, 1, DEFAULTS[a], label=a) for a in ("danceability", "energy")] + [
            gr.Slider(-30, 0, DEFAULTS["loudness"], label="loudness (dB)")
        ]
        sliders += [
            gr.Slider(0, 1, DEFAULTS[a], label=a)
            for a in ("speechiness", "acousticness", "instrumentalness", "valence")
        ]
        sliders += [
            gr.Slider(50, 220, DEFAULTS["tempo"], label="tempo (bpm)"),
            gr.Slider(1, 8, DEFAULTS["duration_min"], label="length (minutes)"),
            gr.Slider(2000, 2020, DEFAULTS["release_year"], label="release year"),
        ]
        btn = gr.Button("Screen it")
        text = gr.Markdown()
        table = gr.Dataframe(label="closest tracks in the catalogue")
        btn.click(handler, [genre, *sliders], [text, table])
    demo.launch()


__all__ = ["AUDIO", "screen", "describe", "launch"]
