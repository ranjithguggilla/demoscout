import numpy as np
import pandas as pd
import pytest

from demoscout import data


@pytest.fixture(scope="session")
def synthetic() -> pd.DataFrame:
    """A small table with the same columns as the real one and a known signal."""
    rng = np.random.default_rng(7)
    n = 900
    genres = np.array(["pop", "rock", "edm", "latin"])
    df = pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(n)],
            "track_name": [f"song {i}" for i in range(n)],
            "track_artist": [f"artist {i % 150}" for i in range(n)],
            "playlist_genre": rng.choice(genres, n),
            "danceability": rng.uniform(0.2, 0.95, n),
            "energy": rng.uniform(0.1, 1, n),
            "key": rng.integers(0, 12, n).astype(float),
            "loudness": rng.uniform(-18, -2, n),
            "mode": rng.integers(0, 2, n).astype(float),
            "speechiness": rng.uniform(0.02, 0.4, n),
            "acousticness": rng.uniform(0, 1, n),
            "instrumentalness": rng.uniform(0, 0.3, n),
            "liveness": rng.uniform(0.05, 0.5, n),
            "valence": rng.uniform(0.05, 0.95, n),
            "tempo": rng.uniform(70, 180, n),
            "duration_ms": rng.uniform(120000, 300000, n),
            "release_year": rng.uniform(2008, 2020, n),
        }
    )
    df["duration_min"] = df["duration_ms"] / 60000
    base = 20 + 2.5 * (df["release_year"] - 2008) + 25 * df["danceability"] + (df["playlist_genre"] == "pop") * 8
    df["track_popularity"] = np.clip(base + rng.normal(0, 8, n), 0, 100).round()
    df["track_album_release_date"] = [f"{int(y)}-06-15" for y in df["release_year"]]
    return df


@pytest.fixture(scope="session")
def split(synthetic):
    return data.artist_split(synthetic)
