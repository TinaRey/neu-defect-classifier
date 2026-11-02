# NEU Surface Defect Classifier

A small PyTorch CNN that classifies steel-surface defects (6 classes) from the
[NEU surface defect database](http://faculty.neu.edu.cn/songkechen/zh_CN/zhuanti/258646/), with
MLflow tracking, a FastAPI inference service, tests, a Dockerfile and CI.

Classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

## Layout
```
configs/default.yaml     all paths and hyperparameters
src/defect/
  config.py              YAML -> Config dataclass
  transforms.py          THE preprocessing function (training, evaluation and API all use it)
  data.py                scan, validate, split, Dataset, DataLoaders
  model.py               SimpleCNN, device selection, checkpoint loading
  train.py               training loop, best checkpoint, MLflow
  evaluate.py            per-class metrics and confusion matrix
  api.py                 FastAPI service
tests/                   pytest (synthetic data only, the dataset is not needed)
Dockerfile, requirements*.txt, .github/workflows/tests.yml
```

## Setup
```bash
uv sync        # creates .venv with Python 3.12 and the pinned dependencies from pyproject.toml / uv.lock
```
Commands below are run through `uv run`, which uses the project environment (no activation needed).
`requirements*.txt` mirror the same pins for the Dockerfile and CI, which use plain pip: when you
change a dependency (`uv add ...`), update them too.
The dataset is **not downloaded automatically**. Place it so that images are at
`<data_dir>/<split>/images/<class>/*.jpg` (the official NEU-DET layout) and set `data_dir` in
`configs/default.yaml` (default `../NEU-DET`). The dataset's own train/validation split is ignored.

## Train and evaluate
```bash
uv run python -m defect.train --config configs/default.yaml                # any config field can be overridden
uv run python -m defect.train --epochs 30 --lr 0.0005 --lr_scheduler none
uv run python -m defect.evaluate --checkpoint artifacts/best.pt            # test-set report
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db                   # browse runs
```
Outputs: `artifacts/best.pt` (weights, class names, image size, train mean/std), `artifacts/splits.json`
(the fixed split, reused by every run), and an MLflow run with params, per-epoch train/val metrics,
test metrics and artifacts.

## Serve
```bash
MODEL_PATH=artifacts/best.pt uv run uvicorn defect.api:app --port 8000
curl -F "file=@image.jpg" localhost:8000/predict
# {"predicted_class":"crazing","confidence":0.993,"probabilities":{"crazing":0.993,...}}
```
`GET /health` returns the status and class list; interactive docs are at `/docs`.

### Docker
```bash
docker build -t neu-defect .
docker run -p 8000:8000 neu-defect
```
The checkpoint `model/best.pt` (about 400 KB) is versioned in the repo and baked into the image. To serve
a different one, mount it and set `MODEL_PATH`, e.g. `-v "$PWD/artifacts/best.pt:/m/best.pt:ro" -e MODEL_PATH=/m/best.pt`.
After retraining, copy `artifacts/best.pt` to `model/best.pt` to update what gets deployed.

### Deploy on Render
`render.yaml` is a Render Blueprint. In the Render dashboard: **New > Blueprint**, connect this GitHub repo,
and apply. Render builds the Dockerfile (a few minutes), injects `PORT`, and checks `/health`.
- The free plan has 512 MB RAM (the API used about 256 MB locally) and spins down after ~15 minutes idle,
  so the first request afterwards takes several seconds while torch loads. Use `plan: starter` to avoid that.
- Try it: `curl -F "file=@image.jpg" https://<your-service>.onrender.com/predict`

## Tests
```bash
uv run pytest -q
```
CI (`.github/workflows/tests.yml`) runs the same command on every push and pull request.

## Design decisions
- **No leakage:** all images are pooled, exact duplicates dropped (one pair exists), then split once
  with a stratified, seeded 70/15/15 split saved to `splits.json`. Normalization mean/std come from
  the train split only. Augmentation (flips) is train-only. The test set is evaluated once, at the end.
- **Same preprocessing everywhere:** `transforms.build_transform` is the only place images are turned
  into tensors; a test checks that the API output equals a direct model call.
- **Model:** 4 x (Conv3x3 -> ReLU -> MaxPool) -> AdaptiveAvgPool -> Linear, about 98k parameters.
  It returns logits; softmax is applied only at inference.
- **Model selection:** best epoch by validation loss (smoother than accuracy on a small validation set);
  the test set plays no part in it.

## Results (seed 42, 50 epochs, cosine LR)
Best validation loss 0.0754 at epoch 48 (val accuracy 96.3%); **test accuracy 94.8%, macro-F1 0.948,
test loss 0.157** on 270 images (95% CI roughly +/-2.6 points). The weakest class is `pitted_surface`
(recall 0.87), mostly confused with `inclusion`.

## Known limitations / ideas for later
- Small test set and a single seed: report mean +/- std over several seeds or k-fold CV.
- Probability calibration is unchecked: treat API probabilities as scores, not true confidences.
- No defect-free class in NEU, so defective-vs-OK recall (a missed defect is the costly error) cannot
  be measured; add defect-free images to measure it.
- Possible accuracy gains: BatchNorm, a wider network, rotation augmentation.
