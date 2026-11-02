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
# Python 3.12; dependency versions are pinned to the ones the model was trained with
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The dataset is **not downloaded automatically**. Place it so that images are at
`<data_dir>/<split>/images/<class>/*.jpg` (the official NEU-DET layout) and set `data_dir` in
`configs/default.yaml` (default `../NEU-DET`). The dataset's own train/validation split is ignored.

## Train and evaluate
```bash
export PYTHONPATH=src            # or: pip install -e .
python -m defect.train --config configs/default.yaml                # any config field can be overridden
python -m defect.train --epochs 30 --lr 0.0005 --lr_scheduler none
python -m defect.evaluate --checkpoint artifacts/best.pt            # test-set report
mlflow ui --backend-store-uri sqlite:///mlflow.db                   # browse runs
```
Outputs: `artifacts/best.pt` (weights, class names, image size, train mean/std), `artifacts/splits.json`
(the fixed split, reused by every run), and an MLflow run with params, per-epoch train/val metrics,
test metrics and artifacts.

## Serve
```bash
MODEL_PATH=artifacts/best.pt uvicorn defect.api:app --port 8000
curl -F "file=@image.jpg" localhost:8000/predict
# {"predicted_class":"crazing","confidence":0.993,"probabilities":{"crazing":0.993,...}}
```
`GET /health` returns the status and class list; interactive docs are at `/docs`.

### Docker
```bash
docker build -t neu-defect .
docker run -p 8000:8000 -v "$PWD/artifacts/best.pt:/models/best.pt:ro" neu-defect
```
The checkpoint is mounted, not baked into the image, so the same image serves any trained model.

## Tests
```bash
python -m pytest -q
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
- **Model selection:** best epoch by validation accuracy; the test set plays no part in it.

## Results (seed 42, 50 epochs, cosine LR)
Best validation accuracy 97.4% (epoch 32); **test accuracy 94.8%, macro-F1 0.948** on 270 images
(95% CI roughly +/-2.6 points). The weakest class is `pitted_surface` (recall 0.84), mostly confused
with `inclusion`.

## Known limitations / ideas for later
- Small test set and a single seed: report mean +/- std over several seeds or k-fold CV.
- Probability calibration is unchecked: treat API probabilities as scores, not true confidences.
- No defect-free class in NEU, so defective-vs-OK recall (a missed defect is the costly error) cannot
  be measured; add defect-free images to measure it.
- Select the best epoch by validation loss instead of accuracy.
- Possible accuracy gains: BatchNorm, a wider network, rotation augmentation.
