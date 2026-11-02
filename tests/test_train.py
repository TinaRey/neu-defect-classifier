import mlflow
import torch

from defect.config import Config
from defect.train import set_seed, train


def tiny_config(tmp_path, data_dir, **kw) -> Config:
    return Config(data_dir=str(data_dir), splits_path=str(tmp_path / "splits.json"), output_dir=str(tmp_path / "out"),
                  val_frac=0.2, test_frac=0.2, batch_size=8, epochs=2, device="cpu",
                  mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db", **kw)


def test_training_saves_checkpoint_and_logs_to_mlflow(tmp_path, fake_dataset, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts to ./mlruns relative to cwd
    cfg = tiny_config(tmp_path, fake_dataset)
    result = train(cfg)

    ckpt = torch.load(result["checkpoint"], map_location="cpu")
    assert ckpt["class_names"] == ["crazing", "inclusion", "patches"]
    assert {"model_state", "mean", "std", "image_size"} <= set(ckpt)

    run = mlflow.search_runs(experiment_names=[cfg.experiment_name]).iloc[0]
    assert run["params.lr"] == str(cfg.lr) and 0 <= run["metrics.test_acc"] <= 1


def test_same_seed_gives_same_results(tmp_path, fake_dataset, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        results.append(train(tiny_config(tmp_path / name, fake_dataset, seed=7)))
    a, b = results
    assert a["best_val_acc"] == b["best_val_acc"] and a["test_acc"] == b["test_acc"]


def test_set_seed_makes_init_reproducible():
    set_seed(1); a = torch.rand(3)
    set_seed(1); b = torch.rand(3)
    assert torch.equal(a, b)
