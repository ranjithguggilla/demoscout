from demoscout import cli, pipeline


def test_cli_fetch_uses_existing_file(tmp_path, monkeypatch, synthetic, capsys):
    path = tmp_path / "x.csv"
    synthetic.to_csv(path, index=False)
    monkeypatch.setattr("demoscout.data.DEFAULT_PATH", path)
    monkeypatch.setattr("demoscout.data.fetch", lambda *a, **k: path)
    assert cli.main(["fetch"]) == 0
    assert str(path) in capsys.readouterr().out


def test_pipeline_exposes_run():
    assert callable(pipeline.run)


def test_full_pipeline_on_small_table(tmp_path, synthetic):
    import json

    import joblib

    csv = tmp_path / "songs.csv"
    synthetic.to_csv(csv, index=False)
    res = pipeline.run(tmp_path / "out", tmp_path / "assets", csv, n_boot=10, ngb_estimators=15)
    assert res["range_selection"]["shipped"] in res["range_selection"]["losses"]
    assert res["hit_selection"]["shipped"] in res["hit_selection"]["losses"]
    assert (tmp_path / "out" / "metrics.json").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["data"]["tracks"] > 800
    assert len(list((tmp_path / "assets").glob("*.png"))) == 5
    scout = joblib.load(tmp_path / "out" / "scout.joblib")
    out = scout.score({"playlist_genre": "pop"})
    assert 0 <= out["hit_probability"] <= 1
