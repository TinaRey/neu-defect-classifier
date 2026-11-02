import numpy as np
import pytest
import torch
from PIL import Image

from defect.data import (DataValidationError, DefectDataset, compute_mean_std, make_splits, scan_images,
                         validate_images)
from defect.transforms import build_transform

from conftest import CLASSES


def test_transform_output_shape_and_dtype():
    img = Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8))
    x = build_transform(200, 0.5, 0.1)(img)
    assert x.shape == (1, 200, 200) and x.dtype == torch.float32


def test_transform_resizes_other_input_sizes():
    img = Image.fromarray(np.zeros((97, 143), dtype=np.uint8))
    assert build_transform(200, 0.5, 0.1)(img).shape == (1, 200, 200)


def test_eval_transform_is_deterministic():
    img = Image.fromarray(np.random.randint(0, 256, (200, 200), dtype=np.uint8))
    t = build_transform(200, 0.5, 0.1, augment=False)
    assert torch.equal(t(img), t(img))


def test_normalization_math():
    img = Image.fromarray(np.full((200, 200), 255, dtype=np.uint8))
    x = build_transform(200, 0.5, 0.25)(img)
    assert torch.allclose(x, torch.full_like(x, 2.0))  # (1.0 - 0.5) / 0.25


def test_scan_and_validate(fake_dataset):
    samples = scan_images(fake_dataset)
    clean, report = validate_images(fake_dataset, samples, 200)
    assert len(clean) == 30 and report["class_counts"] == {c: 10 for c in CLASSES}


def test_validate_drops_duplicates(fake_dataset):
    src = next((fake_dataset / "train/images/crazing").glob("*.jpg"))
    (fake_dataset / "train/images/crazing/copy.jpg").write_bytes(src.read_bytes())
    clean, report = validate_images(fake_dataset, scan_images(fake_dataset), 200)
    assert len(clean) == 30 and len(report["duplicates_dropped"]) == 1


def test_validate_rejects_corrupt_file(fake_dataset):
    (fake_dataset / "train/images/crazing/bad.jpg").write_bytes(b"not an image")
    with pytest.raises(DataValidationError):
        validate_images(fake_dataset, scan_images(fake_dataset), 200)


def test_splits_are_disjoint_stratified_and_reproducible(fake_dataset):
    samples = scan_images(fake_dataset)
    a = make_splits(samples, 0.2, 0.2, seed=1)
    assert a == make_splits(list(reversed(samples)), 0.2, 0.2, seed=1)  # independent of input order
    paths = {k: {p for p, _ in v} for k, v in a.items()}
    assert not (paths["train"] & paths["val"] or paths["train"] & paths["test"] or paths["val"] & paths["test"])
    assert sum(len(v) for v in paths.values()) == len(samples)
    for split in a.values():
        assert {c: sum(l == c for _, l in split) for c in CLASSES} == {c: len(split) // 3 for c in CLASSES}


def test_dataset_item(fake_dataset):
    samples = scan_images(fake_dataset)
    ds = DefectDataset(fake_dataset, samples, CLASSES, build_transform(200, 0.5, 0.2))
    x, y = ds[0]
    assert x.shape == (1, 200, 200) and y == CLASSES.index(samples[0][1])


def test_mean_std_in_valid_range(fake_dataset):
    mean, std = compute_mean_std(fake_dataset, scan_images(fake_dataset), 200)
    assert 0.4 < mean < 0.6 and 0.2 < std < 0.35  # uniform noise: mean ~0.5, std ~0.29
