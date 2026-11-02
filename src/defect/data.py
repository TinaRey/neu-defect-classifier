"""Scanning, validating, splitting and loading the NEU surface-defect images.

Expected layout (the official NEU-DET one; its own train/validation split is
ignored and everything is pooled, then re-split here):

    <data_dir>/<any split>/images/<class_name>/*.jpg
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from defect.config import Config
from defect.transforms import build_raw_transform, build_transform

Sample = tuple[str, str]  # (path relative to data_dir, class name)


class DataValidationError(Exception):
    pass


def scan_images(data_dir: str | Path) -> list[Sample]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DataValidationError(f"data_dir does not exist: {data_dir}")
    paths = sorted(data_dir.glob("*/images/*/*.jpg"))
    if not paths:
        raise DataValidationError(f"No images found under {data_dir}/*/images/<class>/*.jpg")
    return [(str(p.relative_to(data_dir)), p.parent.name) for p in paths]


def validate_images(data_dir: str | Path, samples: list[Sample], image_size: int) -> tuple[list[Sample], dict]:
    """Fail on unreadable files; drop exact-duplicate images (keeping the first).

    Duplicates must go before splitting: otherwise the same pixels could land in
    both train and test, which is leakage.
    """
    data_dir = Path(data_dir)
    seen: dict[str, str] = {}
    clean, duplicates, wrong_size = [], [], []
    for rel, label in samples:
        try:
            with Image.open(data_dir / rel) as img:
                img.load()  # forces a full decode; open() alone only reads the header
                digest = hashlib.md5(img.convert("L").tobytes()).hexdigest()
                if img.size != (image_size, image_size):
                    wrong_size.append(rel)
        except (UnidentifiedImageError, OSError) as e:
            raise DataValidationError(f"Corrupt or unreadable image {rel}: {e}") from e
        if digest in seen:
            duplicates.append((rel, seen[digest]))
            continue
        seen[digest] = rel
        clean.append((rel, label))

    counts = defaultdict(int)
    for _, label in clean:
        counts[label] += 1
    if len(counts) < 2:
        raise DataValidationError(f"Need at least 2 classes, found {sorted(counts)}")
    report = {"n_images": len(clean), "class_counts": dict(sorted(counts.items())),
              "duplicates_dropped": duplicates, "not_resized_natively": len(wrong_size)}
    return clean, report


def make_splits(samples: list[Sample], val_frac: float, test_frac: float, seed: int) -> dict[str, list[Sample]]:
    """Stratified split: each class is shuffled and cut separately so class ratios match in every split."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_class):
        items = sorted(by_class[label])  # sort first so the result doesn't depend on scan order
        rng.shuffle(items)
        n_val, n_test = round(len(items) * val_frac), round(len(items) * test_frac)
        splits["val"] += items[:n_val]
        splits["test"] += items[n_val:n_val + n_test]
        splits["train"] += items[n_val + n_test:]
    paths = [p for split in splits.values() for p, _ in split]
    assert len(paths) == len(set(paths)), "a file appears in more than one split"
    return splits


def load_or_create_splits(cfg: Config) -> dict[str, list[Sample]]:
    """Splits are saved once and reused, so every run and every report uses the same test set."""
    path = Path(cfg.splits_path)
    if path.exists():
        return {k: [tuple(s) for s in v] for k, v in json.loads(path.read_text()).items()}
    samples, report = validate_images(cfg.data_dir, scan_images(cfg.data_dir), cfg.image_size)
    print("Data validation:", json.dumps(report, indent=2))
    splits = make_splits(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))
    return splits


class DefectDataset(Dataset):
    """A Dataset only needs __len__ and __getitem__; DataLoader handles batching and shuffling."""

    def __init__(self, data_dir: str | Path, samples: list[Sample], class_names: list[str], transform):
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rel, label = self.samples[i]
        with Image.open(self.data_dir / rel) as img:
            return self.transform(img), self.class_to_idx[label]


def compute_mean_std(data_dir: str | Path, samples: list[Sample], image_size: int) -> tuple[float, float]:
    """Pixel mean/std over the TRAIN samples only (val/test statistics must not influence preprocessing)."""
    raw = build_raw_transform(image_size)
    total = total_sq = 0.0
    n = 0
    for rel, _ in samples:
        with Image.open(Path(data_dir) / rel) as img:
            x = raw(img).double()
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        n += x.numel()
    mean = total / n
    return mean, (total_sq / n - mean ** 2) ** 0.5


def make_loaders(cfg: Config):
    """Returns ({'train','val','test': DataLoader}, class_names, mean, std)."""
    splits = load_or_create_splits(cfg)
    class_names = sorted({label for _, label in splits["train"]})
    mean, std = compute_mean_std(cfg.data_dir, splits["train"], cfg.image_size)

    g = torch.Generator().manual_seed(cfg.seed)  # makes the shuffle order reproducible
    loaders = {}
    for name, samples in splits.items():
        ds = DefectDataset(cfg.data_dir, samples, class_names,
                           build_transform(cfg.image_size, mean, std, augment=(name == "train")))
        loaders[name] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(name == "train"),
                                   num_workers=cfg.num_workers, generator=g if name == "train" else None)
    return loaders, class_names, mean, std
