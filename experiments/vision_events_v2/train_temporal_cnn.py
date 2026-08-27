#!/usr/bin/env python3
"""Train and evaluate a small temporal CNN on the extracted feature sequences."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = EXPERIMENT_DIR / "artifacts" / "look_away_sequences.npz"
DEFAULT_MODEL = EXPERIMENT_DIR / "artifacts" / "temporal_cnn.pt"
DEFAULT_METRICS = EXPERIMENT_DIR / "artifacts" / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a grouped temporal-CNN pilot.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_count(y: np.ndarray, groups: np.ndarray) -> int:
    unique_groups = len(np.unique(groups))
    class_group_counts = [
        len(np.unique(groups[y == class_value])) for class_value in np.unique(y)
    ]
    return min(5, unique_groups, min(class_group_counts))


def grouped_splits(y: np.ndarray, groups: np.ndarray, seed: int):
    dummy_x = np.zeros((len(y), 1), dtype=np.float32)
    outer_count = split_count(y, groups)
    if outer_count < 2:
        raise RuntimeError("At least two videos containing every class are required.")
    outer = StratifiedGroupKFold(n_splits=outer_count, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(dummy_x, y, groups=groups))

    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]
    inner_count = split_count(y_train_val, groups_train_val)
    if inner_count < 2:
        raise RuntimeError("The non-test videos cannot form grouped train and validation splits.")
    inner = StratifiedGroupKFold(n_splits=inner_count, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(
            np.zeros((len(train_val_idx), 1), dtype=np.float32),
            y_train_val,
            groups=groups_train_val,
        )
    )
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def fit_preprocessor(X: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(all="ignore"):
        median = np.nanmedian(X, axis=(0, 1))
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    imputed = np.where(np.isfinite(X), X, median.reshape(1, 1, -1))
    mean = imputed.mean(axis=(0, 1)).astype(np.float32)
    std = imputed.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return {"median": median, "mean": mean, "std": std}


def transform(X: np.ndarray, preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    median = preprocessor["median"].reshape(1, 1, -1)
    mean = preprocessor["mean"].reshape(1, 1, -1)
    std = preprocessor["std"].reshape(1, 1, -1)
    imputed = np.where(np.isfinite(X), X, median)
    return ((imputed - mean) / std).astype(np.float32)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence.transpose(1, 2))
        pooled = torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=1)
        return self.classifier(pooled)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all, predictions_all, probabilities_all = [], [], []
    for features, labels in loader:
        logits = model(features.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        labels_all.append(labels.numpy())
        predictions_all.append(probabilities.argmax(axis=1))
        probabilities_all.append(probabilities)
    return (
        np.concatenate(labels_all),
        np.concatenate(predictions_all),
        np.concatenate(probabilities_all),
    )


def describe_split(name: str, indices: np.ndarray, y: np.ndarray, groups: np.ndarray) -> str:
    counts = np.bincount(y[indices], minlength=len(np.unique(y)))
    return f"{name}: rows={len(indices)}, videos={len(np.unique(groups[indices]))}, classes={counts.tolist()}"


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise SystemExit("Epochs, patience, and batch size must be positive.")
    set_seed(args.seed)
    device = select_device(args.device)

    with np.load(args.data, allow_pickle=False) as dataset:
        X = dataset["X"].astype(np.float32)
        y = dataset["y"].astype(np.int64)
        groups = dataset["groups"].astype(str)
        window_ids = dataset["window_ids"].astype(str)
        feature_names = dataset["feature_names"].astype(str).tolist()
        metadata = json.loads(str(dataset["metadata"].item()))

    labels = sorted(np.unique(y).tolist())
    if labels != list(range(len(labels))):
        raise RuntimeError(f"Class labels must be consecutive integers starting at zero; got {labels}.")
    target_names = metadata["target_names"]
    train_idx, val_idx, test_idx = grouped_splits(y, groups, args.seed)
    print(f"Device: {device} | sequences: {X.shape} | target: {metadata['target']}")
    print(describe_split("train", train_idx, y, groups))
    print(describe_split("validation", val_idx, y, groups))
    print(describe_split("test", test_idx, y, groups))

    preprocessor = fit_preprocessor(X[train_idx])
    X_train = transform(X[train_idx], preprocessor)
    X_val = transform(X[val_idx], preprocessor)
    X_test = transform(X[test_idx], preprocessor)
    train_loader = make_loader(X_train, y[train_idx], args.batch_size, True)
    val_loader = make_loader(X_val, y[val_idx], args.batch_size, False)
    test_loader = make_loader(X_test, y[test_idx], args.batch_size, False)

    n_classes = len(labels)
    model_config = {"n_features": X.shape[2], "n_classes": n_classes, "dropout": 0.3}
    model = TemporalCNN(**model_config).to(device)
    class_counts = np.bincount(y[train_idx], minlength=n_classes)
    class_weights = len(train_idx) / (n_classes * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    best_state = None
    best_epoch = 0
    best_val_f1 = -np.inf
    without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_true, val_pred, _ = predict(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": val_f1})
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_loss={loss:.4f} val_macro_f1={val_f1:.3f}")
        if without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    test_true, test_pred, test_probabilities = predict(model, test_loader, device)
    macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(test_true, test_pred)
    report_text = classification_report(
        test_true,
        test_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        test_true,
        test_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(test_true, test_pred, labels=labels)
    print(f"Best epoch: {best_epoch} | validation macro F1: {best_val_f1:.3f}")
    print(report_text)
    print(f"Test macro F1: {macro_f1:.3f}")
    print(f"Test balanced accuracy: {balanced_accuracy:.3f}")
    print("Confusion matrix:")
    print(matrix)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": model_config,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metadata": metadata,
        "target_names": target_names,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_val_f1,
        "test_videos": sorted(np.unique(groups[test_idx]).tolist()),
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": balanced_accuracy,
    }
    torch.save(checkpoint, args.model_output)

    metrics = {
        "data": str(args.data),
        "model": str(args.model_output),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_val_f1,
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
        "train_videos": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_videos": sorted(np.unique(groups[val_idx]).tolist()),
        "test_videos": sorted(np.unique(groups[test_idx]).tolist()),
        "test_predictions": [
            {
                "window_id": window_ids[index],
                "actual": int(actual),
                "predicted": int(predicted),
                "probabilities": probabilities.tolist(),
            }
            for index, actual, predicted, probabilities in zip(
                test_idx, test_true, test_pred, test_probabilities
            )
        ],
        "history": history,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {args.model_output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()

