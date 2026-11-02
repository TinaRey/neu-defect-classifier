"""Train SimpleCNN, keep the best checkpoint by validation loss, track with MLflow.

Usage:
    python -m defect.train --config configs/default.yaml --epochs 30 --lr 0.0005
"""
import argparse
import random
from dataclasses import asdict, fields
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from defect.config import Config, load_config
from defect.data import make_loaders
from defect.evaluate import evaluate_model, format_report
from defect.model import SimpleCNN, pick_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all accelerator RNGs, including weight initialization


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)  # train() / eval() switch layers like dropout and batchnorm (none yet, but safe habit)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):  # no gradient graph is built during evaluation: faster, less memory
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()  # gradients accumulate by default, so clear the previous batch's
                loss.backward()        # backpropagation: fills p.grad for every parameter
                optimizer.step()       # updates parameters using p.grad
            total_loss += loss.item() * len(y)  # loss is a per-batch mean; weight by size for an exact epoch mean
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    loaders, class_names, mean, std = make_loaders(cfg)

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_scheduler == "cosine":
        # Smoothly lowers lr from cfg.lr to ~0 over all epochs: big steps early, fine-tuning late.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler!r} (use 'cosine' or 'none')")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(asdict(cfg))
        mlflow.log_params({"resolved_device": str(device), "n_parameters": sum(p.numel() for p in model.parameters()),
                           **{f"n_{k}": len(v.dataset) for k, v in loaders.items()}})

        best_val_loss, val_acc_at_best = float("inf"), 0.0
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
            val_loss, val_acc = run_epoch(model, loaders["val"], loss_fn, device)
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                                "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            if scheduler:
                scheduler.step()  # once per epoch, after the optimizer has stepped through all batches
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

            # Select on validation loss: smoother than accuracy on a small val set (270 images), it uses
            # the predicted probabilities, and it rarely ties. VALIDATION only; test is untouched until the end.
            if val_loss < best_val_loss:
                best_val_loss, val_acc_at_best = val_loss, val_acc
                # Everything inference needs lives in one file: weights, class order, preprocessing stats.
                torch.save({"model_state": model.state_dict(), "class_names": class_names,
                            "mean": mean, "std": std, "image_size": cfg.image_size,
                            "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc, "config": asdict(cfg)}, ckpt_path)

        # Reload the best weights (not the last epoch's) and evaluate on test exactly once.
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc = run_epoch(model, loaders["test"], loss_fn, device)
        report = evaluate_model(model, loaders["test"], class_names, device)
        mlflow.log_metrics({"best_val_loss": best_val_loss, "val_acc_at_best": val_acc_at_best, "test_loss": test_loss, "test_acc": test_acc,
                            "test_macro_f1": report["macro_f1"],
                            **{f"test_recall_{c}": m["recall"] for c, m in report["per_class"].items()}})
        mlflow.log_dict(report, "test_report.json")
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact(cfg.splits_path)
        print(format_report(report))
        print(f"best epoch {ckpt['epoch']} (val loss {best_val_loss:.4f}, acc {val_acc_at_best:.3f}) | test loss {test_loss:.4f} acc {test_acc:.3f}")

    return {"best_val_loss": best_val_loss, "val_acc_at_best": val_acc_at_best, "test_acc": test_acc, "checkpoint": str(ckpt_path)}


def parse_args() -> tuple[str, dict]:
    """--config PATH plus one optional --<field> VALUE flag per Config field, typed from the dataclass."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    for f in fields(Config):
        p.add_argument(f"--{f.name}", type=type(f.default))
    args = vars(p.parse_args())
    config_path = args.pop("config")
    return config_path, {k: v for k, v in args.items() if v is not None}


if __name__ == "__main__":
    config_path, overrides = parse_args()
    train(load_config(config_path, overrides))
