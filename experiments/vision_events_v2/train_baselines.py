#!/usr/bin/env python3
"""Train non-temporal baselines on summaries of the same extracted sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_temporal_cnn import grouped_splits


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = EXPERIMENT_DIR / "artifacts" / "look_away_sequences.npz"
DEFAULT_MODEL = EXPERIMENT_DIR / "artifacts" / "summary_random_forest.joblib"
DEFAULT_METRICS = EXPERIMENT_DIR / "artifacts" / "baseline_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped summary-feature baselines.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarise_sequences(X: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.concatenate(
            [
                np.nanmean(X, axis=1),
                np.nanstd(X, axis=1),
                np.nanmin(X, axis=1),
                np.nanmax(X, axis=1),
            ],
            axis=1,
        ).astype(np.float32)


def main() -> None:
    args = parse_args()
    with np.load(args.data, allow_pickle=False) as dataset:
        X = dataset["X"].astype(np.float32)
        y = dataset["y"].astype(np.int64)
        groups = dataset["groups"].astype(str)
        feature_names = dataset["feature_names"].astype(str).tolist()
        metadata = json.loads(str(dataset["metadata"].item()))

    summary_X = summarise_sequences(X)
    summary_names = [
        f"{statistic}_{feature}"
        for statistic in ("mean", "std", "min", "max")
        for feature in feature_names
    ]
    train_idx, val_idx, test_idx = grouped_splits(y, groups, args.seed)
    target_names = metadata["target_names"]

    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(summary_X[train_idx], y[train_idx])
        prediction = model.predict(summary_X[test_idx])
        result = {
            "macro_f1": f1_score(y[test_idx], prediction, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], prediction),
            "confusion_matrix": confusion_matrix(y[test_idx], prediction).tolist(),
            "classification_report": classification_report(
                y[test_idx],
                prediction,
                labels=list(range(len(target_names))),
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            ),
        }
        fitted_models[name] = model
        results[name] = result
        print(
            f"{name}: macro_f1={result['macro_f1']:.3f}, "
            f"balanced_accuracy={result['balanced_accuracy']:.3f}"
        )
        print(np.asarray(result["confusion_matrix"]))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": fitted_models["random_forest"],
            "summary_feature_names": summary_names,
            "source_feature_names": feature_names,
            "metadata": metadata,
            "test_videos": sorted(np.unique(groups[test_idx]).tolist()),
        },
        args.model_output,
    )
    metrics = {
        "seed": args.seed,
        "train_videos": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_videos": sorted(np.unique(groups[val_idx]).tolist()),
        "test_videos": sorted(np.unique(groups[test_idx]).tolist()),
        "results": results,
    }
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved baseline model: {args.model_output}")
    print(f"Saved baseline metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()

