"""Loading, cleaning and splitting the Spotify songs table."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-01-21/spotify_songs.csv"
DEFAULT_PATH = Path("data/spotify_songs.csv")

AUDIO = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
NUMERIC = AUDIO + ["duration_min", "release_year"]
CATEGORICAL = ["playlist_genre"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "track_popularity"
HIT_THRESHOLD = 65


def fetch(path: Path = DEFAULT_PATH, url: str = RAW_URL) -> Path:
    """Download the csv if it is not already on disk."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https url
    return path


def load_raw(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(fetch(path))


def clean(raw: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """Drop rows without an artist, parse release dates, optionally collapse repeats.

    The same track id shows up once per playlist it sits on, so the raw file
    has 32,833 rows for 28,356 tracks. With ``dedupe`` the first row per id wins.
    """
    df = raw.dropna(subset=["track_artist", "track_name"]).copy()
    if dedupe:
        df = df.drop_duplicates("track_id", keep="first")
    released = pd.to_datetime(df["track_album_release_date"], errors="coerce", format="mixed")
    df = df.loc[released.notna()].copy()
    released = released.loc[df.index]
    df["release_date"] = released
    df["release_year"] = released.dt.year + (released.dt.dayofyear - 1) / 365.25
    df["duration_min"] = df["duration_ms"] / 60000.0
    return df.reset_index(drop=True)


def artist_bucket(artist: str, seed: int = 0) -> float:
    """Stable value in [0, 1) for an artist name, so splits survive row reordering."""
    digest = hashlib.md5(f"{seed}:{artist}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def artist_split(df: pd.DataFrame, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 0) -> pd.Series:
    """Label rows train / val / test so that no artist appears in two parts."""
    cut1 = fractions[0]
    cut2 = fractions[0] + fractions[1]
    u = df["track_artist"].map(lambda a: artist_bucket(a, seed))
    labels = np.where(u < cut1, "train", np.where(u < cut2, "val", "test"))
    return pd.Series(labels, index=df.index, name="split")


def random_split(df: pd.DataFrame, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2), seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    cut1 = fractions[0]
    cut2 = fractions[0] + fractions[1]
    labels = np.where(u < cut1, "train", np.where(u < cut2, "val", "test"))
    return pd.Series(labels, index=df.index, name="split")


def forward_split(df: pd.DataFrame, train_end: float = 2018.5, val_end: float = 2019.4) -> pd.Series:
    """Train on older releases, validate and test on newer ones."""
    y = df["release_year"]
    labels = np.where(y < train_end, "train", np.where(y < val_end, "val", "test"))
    return pd.Series(labels, index=df.index, name="split")


def hit_label(popularity: pd.Series | np.ndarray, threshold: int = HIT_THRESHOLD) -> np.ndarray:
    return (np.asarray(popularity) >= threshold).astype(int)
