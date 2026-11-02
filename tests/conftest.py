import numpy as np
import pytest
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches"]


@pytest.fixture
def fake_dataset(tmp_path):
    """Tiny synthetic dataset in the NEU-DET layout: 10 unique RGB-gray images per class."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 6), ("validation", 4)]:
        for c in CLASSES:
            d = tmp_path / split / "images" / c
            d.mkdir(parents=True)
            for i in range(n):
                gray = rng.integers(0, 256, (200, 200), dtype=np.uint8)
                Image.fromarray(np.stack([gray] * 3, axis=-1)).save(d / f"{c}_{split}_{i}.jpg")
    return tmp_path
