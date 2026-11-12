import numpy as np
import pandas as pd

from demoscout import data


def test_artist_split_never_shares_artists(synthetic):
    s = data.artist_split(synthetic)
    groups = {k: set(synthetic.loc[s == k, "track_artist"]) for k in ("train", "val", "test")}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]
    assert set(s) == {"train", "val", "test"}


def test_artist_split_is_stable_under_row_order(synthetic):
    a = data.artist_split(synthetic)
    shuffled = synthetic.sample(frac=1, random_state=3)
    b = data.artist_split(shuffled)
    assert (a.loc[b.index] == b).all()


def test_random_split_fractions(synthetic):
    s = data.random_split(synthetic)
    assert abs((s == "train").mean() - 0.6) < 0.06


def test_forward_split_orders_by_date(synthetic):
    s = data.forward_split(synthetic)
    assert synthetic.loc[s == "train", "release_year"].max() < synthetic.loc[s == "test", "release_year"].min()


def test_clean_dedupes_and_parses_dates():
    raw = pd.DataFrame(
        {
            "track_id": ["a", "a", "b", "c"],
            "track_name": ["x", "x", "y", None],
            "track_artist": ["p", "p", "q", "r"],
            "track_album_release_date": ["2019-06-14", "2019-06-14", "2015", "2018-01-01"],
            "duration_ms": [180000, 180000, 240000, 200000],
        }
    )
    out = data.clean(raw)
    assert list(out["track_id"]) == ["a", "b"]
    assert out["release_year"].iloc[0] > 2019.4
    assert np.isclose(out["duration_min"].iloc[1], 4.0)
    assert len(data.clean(raw, dedupe=False)) == 3


def test_hit_label_threshold():
    assert list(data.hit_label([10, 64, 65, 90])) == [0, 0, 1, 1]
