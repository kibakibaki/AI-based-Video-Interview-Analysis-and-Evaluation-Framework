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
MIN_FACE_VISIBILITY = 0.5
MIN_VISUAL_OBSERVABILITY = 0.5
MIN_THRESHOLD_COVERAGE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped summary-feature baselines.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-validation-coverage",
        type=float,
        default=MIN_THRESHOLD_COVERAGE,
        help="Minimum share of observable validation windows that must receive a binary state.",
    )
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


def select_abstention_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_coverage: float = MIN_THRESHOLD_COVERAGE,
) -> dict[str, float | str]:
    """Choose a prediction-probability gate using validation labels only.

    Macro F1 is maximised subject to a minimum coverage constraint, preventing
    the degenerate solution of abstaining from nearly every difficult window.
    """
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    candidates = np.unique(np.concatenate(([0.5], confidences)))
    best = None

    for threshold in candidates:
        accepted = confidences >= threshold
        coverage = float(np.mean(accepted))
        if coverage < min_coverage or len(np.unique(y_true[accepted])) < 2:
            continue
        macro_f1 = float(
            f1_score(
                y_true[accepted],
                predictions[accepted],
                average="macro",
                zero_division=0,
            )
        )
        ranking = (macro_f1, coverage, -float(threshold))
        if best is None or ranking > best[0]:
            best = (ranking, float(threshold), coverage, macro_f1)

    if best is None:
        return {
            "min_prediction_probability": 0.5,
            "validation_coverage": 1.0,
            "validation_macro_f1_on_classified": 0.0,
            "minimum_validation_coverage": min_coverage,
            "threshold_source": "validation_max_macro_f1_with_coverage_fallback",
        }

    _, threshold, coverage, macro_f1 = best
    return {
        "min_prediction_probability": threshold,
        "validation_coverage": coverage,
        "validation_macro_f1_on_classified": macro_f1,
        "minimum_validation_coverage": min_coverage,
        "threshold_source": "validation_max_macro_f1_with_coverage",
    }


def decision_state(
    predicted: int,
    probability: float,
    face_visibility: float,
    visual_observability: float,
    threshold: float,
    target_names: list[str],
) -> str:
    if (
        face_visibility < MIN_FACE_VISIBILITY
        or visual_observability < MIN_VISUAL_OBSERVABILITY
    ):
        return "unobservable"
    if probability < threshold:
        return "uncertain"
    return target_names[predicted]


def main() -> None:
    args = parse_args()
    if not 0 < args.min_validation_coverage <= 1:
        raise SystemExit("--min-validation-coverage must be in (0, 1].")
    with np.load(args.data, allow_pickle=False) as dataset:
        X = dataset["X"].astype(np.float32)
        y = dataset["y"].astype(np.int64)
        groups = dataset["groups"].astype(str)
        window_ids = dataset["window_ids"].astype(str)
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
    decision_policies = {}
    face_index = feature_names.index("face_detected")
    face_visibility = np.mean(X[:, :, face_index], axis=1)
    required_observable_indices = [
        feature_names.index(name)
        for name in (
            "iris_horizontal_mean",
            "iris_vertical_mean",
            "head_pitch",
            "head_yaw",
        )
    ]
    visual_observability = np.mean(
        np.all(np.isfinite(X[:, :, required_observable_indices]), axis=2),
        axis=1,
    )
    for name, model in models.items():
        model.fit(summary_X[train_idx], y[train_idx])
        validation_probabilities = model.predict_proba(summary_X[val_idx])
        visible_validation = (
            (face_visibility[val_idx] >= MIN_FACE_VISIBILITY)
            & (visual_observability[val_idx] >= MIN_VISUAL_OBSERVABILITY)
        )
        threshold_result = select_abstention_threshold(
            y[val_idx][visible_validation],
            validation_probabilities[visible_validation],
            min_coverage=args.min_validation_coverage,
        )
        decision_policy = {
            "min_face_visibility": MIN_FACE_VISIBILITY,
            "min_visual_observability": MIN_VISUAL_OBSERVABILITY,
            **threshold_result,
        }
        test_probabilities = model.predict_proba(summary_X[test_idx])
        prediction = test_probabilities.argmax(axis=1)
        prediction_probabilities = test_probabilities.max(axis=1)
        test_states = [
            decision_state(
                int(predicted),
                float(probability),
                float(visibility),
                float(observability),
                float(decision_policy["min_prediction_probability"]),
                target_names,
            )
            for predicted, probability, visibility, observability in zip(
                prediction,
                prediction_probabilities,
                face_visibility[test_idx],
                visual_observability[test_idx],
            )
        ]
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
            "decision_policy": decision_policy,
            "test_decision_counts": {
                state: test_states.count(state)
                for state in (*target_names, "uncertain", "unobservable")
            },
            "test_predictions": [
                {
                    "window_id": str(window_ids[index]),
                    "video": str(groups[index]),
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "state": state,
                    "face_visibility_ratio": round(float(visibility), 6),
                    "visual_observability_ratio": round(float(observability), 6),
                    "prediction_probability": round(float(probability), 6),
                    "probabilities": {
                        target_name: round(float(class_probability), 6)
                        for target_name, class_probability in zip(
                            target_names,
                            class_probabilities,
                        )
                    },
                }
                for index, actual, predicted, state, visibility, observability, probability, class_probabilities
                in zip(
                    test_idx,
                    y[test_idx],
                    prediction,
                    test_states,
                    face_visibility[test_idx],
                    visual_observability[test_idx],
                    prediction_probabilities,
                    test_probabilities,
                )
            ],
        }
        fitted_models[name] = model
        decision_policies[name] = decision_policy
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
            "decision_policy": decision_policies["random_forest"],
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
