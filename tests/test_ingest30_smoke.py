from isaac_datagen.ingest30_smoke import smoke_tally


def test_smoke_tally(tmp_path):
    for cls, n in [("cereal001", 6), ("flour001", 0)]:
        obs = tmp_path / cls / "render000" / "obs"
        obs.mkdir(parents=True)
        for i in range(n):
            (obs / f"obs_{i:04d}.png").write_bytes(b"x")
    (tmp_path / "flour001" / "render000").mkdir(exist_ok=True)
    rows = smoke_tally(tmp_path)
    assert rows == [("cereal001", 6), ("flour001", 0)]
