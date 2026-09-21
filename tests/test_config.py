from dcmseg.config import load_config


def test_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epochs: 10\n  lr: 0.001\n")
    cfg = load_config(path, ["train.epochs=3", "model.name=dynunet", "data.patch_size=[32, 32, 16]"])
    assert cfg["train"]["epochs"] == 3
    assert cfg["train"]["lr"] == 0.001
    assert cfg["model"]["name"] == "dynunet"
    assert cfg["data"]["patch_size"] == [32, 32, 16]
