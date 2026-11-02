"""Inference API.

Run:  MODEL_PATH=artifacts/best.pt uvicorn defect.api:app --port 8000
Call: curl -F "file=@some_image.jpg" localhost:8000/predict
"""
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from defect.model import SimpleCNN
from defect.transforms import build_transform

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def load_model(path: str):
    """Rebuild everything inference needs from the single checkpoint file written by training."""
    ckpt = torch.load(path, map_location="cpu")
    model = SimpleCNN(num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # inference mode for layers; we also disable gradients at call time
    # Same function training used, with the TRAIN-set mean/std saved in the checkpoint.
    transform = build_transform(ckpt["image_size"], ckpt["mean"], ckpt["std"])
    return model, ckpt["class_names"], transform


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request. Path is read here (not at import) so tests can set it.
    app.state.model, app.state.class_names, app.state.transform = load_model(
        os.environ.get("MODEL_PATH", "artifacts/best.pt"))
    yield


app = FastAPI(title="NEU surface defect classifier", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "classes": request.app.state.class_names}


@app.post("/predict", response_model=Prediction)
def predict(request: Request, file: UploadFile = File(...)):
    # A plain `def` endpoint runs in a worker thread, so CPU-bound inference doesn't block the event loop.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # decode fully now so a truncated file fails here, not later
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Uploaded file is not a valid image")

    state = request.app.state
    x = state.transform(image).unsqueeze(0)  # add the batch dimension: (1, 1, H, W)
    with torch.inference_mode():
        probs = torch.softmax(state.model(x), dim=1)[0]
    probabilities = dict(zip(state.class_names, probs.tolist()))
    top = probs.argmax().item()
    return Prediction(predicted_class=state.class_names[top], confidence=probs[top].item(),
                      probabilities=probabilities)
