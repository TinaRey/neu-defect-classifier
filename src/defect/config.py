from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    data_dir: str = "../NEU-DET"
    splits_path: str = "artifacts/splits.json"
    output_dir: str = "artifacts"

    image_size: int = 200
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    lr_scheduler: str = "cosine"  # cosine | none
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "neu-defect-classification"


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Read a YAML file into a Config. Unknown keys raise, so typos are not silent."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return replace(Config(), **raw)
