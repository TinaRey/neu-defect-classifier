"""Per-class test metrics and confusion matrix, in plain PyTorch (no scikit-learn).

Possible improvements (deliberately not done yet):
  - The test set is small (45 images/class): 1 error = ~0.4 points, 95% CI about +/-2.6 points. Report
    mean +/- std over several seeds or k-fold cross-validation instead of one split.
  - Calibration is unchecked (reliability curve / expected calibration error), so API probabilities
    should be read as scores, not true confidences.
  - NEU has no "defect-free" class, so defective-vs-OK recall (the costly error in a factory,
    a missed defect) cannot be measured. Add defect-free images to measure it.

Usage:
    python -m defect.evaluate --config configs/default.yaml --checkpoint artifacts/best.pt
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from defect.config import load_config
from defect.data import DefectDataset, load_or_create_splits
from defect.model import SimpleCNN, pick_device
from defect.transforms import build_transform


@torch.no_grad()  # no gradients needed to evaluate
def predict_all(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """cm[i, j] = number of images of true class i predicted as class j."""
    # Encode each (true, pred) pair as one integer, count occurrences, reshape to a square.
    return torch.bincount(targets * num_classes + preds, minlength=num_classes ** 2).view(num_classes, num_classes)


def evaluate_model(model, loader, class_names: list[str], device) -> dict:
    preds, targets = predict_all(model, loader, device)
    cm = confusion_matrix(preds, targets, len(class_names))
    diag = cm.diag().float()
    precision = diag / cm.sum(dim=0).clamp(min=1)  # of images predicted as c, how many truly are c
    recall = diag / cm.sum(dim=1).clamp(min=1)     # of images truly c, how many were found
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "accuracy": (diag.sum() / cm.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class": {c: {"precision": precision[i].item(), "recall": recall[i].item(), "f1": f1[i].item(),
                          "support": int(cm[i].sum())} for i, c in enumerate(class_names)},
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }


def format_report(report: dict) -> str:
    names = report["class_names"]
    w = max(len(n) for n in names) + 2
    lines = [f"{'':{w}}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}"]
    for n, m in report["per_class"].items():
        lines.append(f"{n:{w}}{m['precision']:10.3f}{m['recall']:8.3f}{m['f1']:8.3f}{m['support']:9d}")
    lines.append(f"\naccuracy {report['accuracy']:.4f} | macro-F1 {report['macro_f1']:.4f}")
    lines.append("\nconfusion matrix (rows = true, columns = predicted):")
    lines.append(f"{'':{w}}" + "".join(f"{n[:7]:>8}" for n in names))
    for n, row in zip(names, report["confusion_matrix"]):
        lines.append(f"{n:{w}}" + "".join(f"{v:8d}" for v in row))
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="artifacts/best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SimpleCNN(num_classes=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Use the preprocessing stats stored with the checkpoint, exactly as inference will.
    samples = load_or_create_splits(cfg)[args.split]
    ds = DefectDataset(cfg.data_dir, samples, ckpt["class_names"],
                       build_transform(ckpt["image_size"], ckpt["mean"], ckpt["std"]))
    report = evaluate_model(model, DataLoader(ds, batch_size=cfg.batch_size), ckpt["class_names"], device)
    print(format_report(report))
    out = f"{cfg.output_dir}/{args.split}_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out}")
