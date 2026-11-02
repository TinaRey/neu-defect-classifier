import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from defect.api import app
from defect.model import SimpleCNN
from defect.transforms import build_transform

CLASSES = ["crazing", "inclusion", "patches"]
MEAN, STD = 0.5, 0.2


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App backed by a tiny randomly initialised checkpoint, so tests need neither data nor training."""
    torch.manual_seed(0)
    model = SimpleCNN(num_classes=len(CLASSES))
    path = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "class_names": CLASSES,
                "mean": MEAN, "std": STD, "image_size": 200}, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(app) as c:  # the `with` runs the startup (lifespan) code that loads the model
        c.reference_model = model.eval()
        yield c


def image_bytes(fmt="PNG", mode="L", size=(200, 200), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3 if mode == "RGB" else 1), dtype=np.uint8)
    img = Image.fromarray(arr.squeeze(), mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def post(client, data: bytes, name="x.png"):
    return client.post("/predict", files={"file": (name, data, "image/png")})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["classes"] == CLASSES


def test_predict_response_structure(client):
    r = post(client, image_bytes())
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert body["predicted_class"] == max(body["probabilities"], key=body["probabilities"].get)


def test_api_matches_direct_model_call(client):
    """Training-time preprocessing + model applied directly must equal what the API returns."""
    data = image_bytes(seed=3)
    x = build_transform(200, MEAN, STD)(Image.open(io.BytesIO(data))).unsqueeze(0)
    with torch.no_grad():
        expected = torch.softmax(client.reference_model(x), dim=1)[0].tolist()
    got = post(client, data).json()["probabilities"]
    assert [got[c] for c in CLASSES] == pytest.approx(expected, abs=1e-5)


def test_formats_and_modes_accepted(client):
    for fmt, mode in [("JPEG", "L"), ("JPEG", "RGB"), ("PNG", "RGB"), ("BMP", "L")]:
        assert post(client, image_bytes(fmt, mode)).status_code == 200, (fmt, mode)


def test_other_image_sizes_are_resized(client):
    assert post(client, image_bytes(size=(97, 143))).status_code == 200


def test_invalid_file_rejected(client):
    assert post(client, b"definitely not an image").status_code == 400


def test_missing_file_rejected(client):
    assert client.post("/predict").status_code == 422


def test_oversized_upload_rejected(client):
    assert post(client, b"0" * (5 * 1024 * 1024 + 10)).status_code == 413
