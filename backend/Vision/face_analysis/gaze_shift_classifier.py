"""Binary gaze-shift inference with explicit uncertain/unobservable outcomes."""

from __future__ import annotations

import os
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from .gaze_shift_features import FEATURE_NAMES, extract_frame_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "vision_events_v2"
    / "artifacts"
    / "summary_random_forest.joblib"
)
TARGET_NAMES = ("no_gaze_shift", "gaze_shift")
DEFAULT_MIN_FACE_VISIBILITY = 0.5
DEFAULT_MIN_VISUAL_OBSERVABILITY = 0.5
DEFAULT_MIN_PREDICTION_PROBABILITY = 0.65


class GazeShiftModelUnavailable(RuntimeError):
    """Raised when the trained binary model cannot be loaded."""


class GazeShiftModelContractError(RuntimeError):
    """Raised when a model does not match the frozen binary input contract."""


def summarise_sequence(sequence: np.ndarray) -> np.ndarray:
    """Match the mean/std/min/max summary used by the RF training pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.concatenate(
            [
                np.nanmean(sequence, axis=0),
                np.nanstd(sequence, axis=0),
                np.nanmin(sequence, axis=0),
                np.nanmax(sequence, axis=0),
            ]
        ).astype(np.float32)


class GazeShiftClassifier:
    """Apply the observable-quality gate before binary model inference."""

    def __init__(self, payload: dict, model_name: str = "summary_random_forest.joblib"):
        self.model = payload.get("model")
        self.feature_names = tuple(payload.get("source_feature_names", ()))
        self.metadata = payload.get("metadata", {})
        self.target_names = tuple(self.metadata.get("target_names", ()))
        self.model_name = model_name

        if self.model is None:
            raise GazeShiftModelContractError("Model payload has no classifier.")
        if self.feature_names != FEATURE_NAMES:
            raise GazeShiftModelContractError(
                "Model feature order does not match the frozen V2 feature contract."
            )
        if self.metadata.get("target") != "binary" or self.target_names != TARGET_NAMES:
            raise GazeShiftModelContractError(
                "Only the frozen no_gaze_shift/gaze_shift binary target is supported."
            )

        classes = tuple(int(value) for value in getattr(self.model, "classes_", ()))
        if classes != (0, 1):
            raise GazeShiftModelContractError(
                f"Binary model classes must be [0, 1]; received {classes}."
            )

        policy = payload.get("decision_policy", {})
        self.policy_metadata = dict(policy)
        self.min_face_visibility = float(
            policy.get("min_face_visibility", DEFAULT_MIN_FACE_VISIBILITY)
        )
        self.min_prediction_probability = float(
            policy.get(
                "min_prediction_probability",
                DEFAULT_MIN_PREDICTION_PROBABILITY,
            )
        )
        self.min_visual_observability = float(
            policy.get(
                "min_visual_observability",
                DEFAULT_MIN_VISUAL_OBSERVABILITY,
            )
        )
        self.threshold_source = policy.get(
            "threshold_source",
            "fallback_default_for_legacy_artifact",
        )
        self.sample_rate_hz = float(self.metadata.get("sample_rate_hz", 10.0))
        self.window_seconds = float(self.metadata.get("window_seconds", 3.0))
        self.sequence_length = int(
            self.metadata.get(
                "sequence_length",
                round(self.sample_rate_hz * self.window_seconds),
            )
        )

    @property
    def decision_policy(self) -> dict:
        return {
            **self.policy_metadata,
            "min_face_visibility": self.min_face_visibility,
            "min_visual_observability": self.min_visual_observability,
            "min_prediction_probability": self.min_prediction_probability,
            "threshold_source": self.threshold_source,
        }

    def predict_sequence(self, sequence: np.ndarray) -> dict:
        expected_shape = (self.sequence_length, len(self.feature_names))
        if sequence.shape != expected_shape:
            raise GazeShiftModelContractError(
                f"Expected sequence shape {expected_shape}; received {sequence.shape}."
            )

        face_index = self.feature_names.index("face_detected")
        face_visibility = float(np.mean(sequence[:, face_index]))
        required_indices = [
            self.feature_names.index(name)
            for name in (
                "iris_horizontal_mean",
                "iris_vertical_mean",
                "head_pitch",
                "head_yaw",
            )
        ]
        visual_observability = float(
            np.mean(np.all(np.isfinite(sequence[:, required_indices]), axis=1))
        )
        base_result = {
            "face_visibility_ratio": round(face_visibility, 4),
            "visual_observability_ratio": round(visual_observability, 4),
            "prediction_probability": None,
            "probabilities": {
                "no_gaze_shift": None,
                "gaze_shift": None,
            },
        }

        if face_visibility < self.min_face_visibility:
            return {
                **base_result,
                "state": "unobservable",
                "reason": "insufficient_face_visibility",
            }
        if visual_observability < self.min_visual_observability:
            return {
                **base_result,
                "state": "unobservable",
                "reason": "insufficient_eye_or_pose_observability",
            }

        summary = summarise_sequence(sequence).reshape(1, -1)
        probabilities = np.asarray(self.model.predict_proba(summary)[0], dtype=float)
        predicted_index = int(np.argmax(probabilities))
        prediction_probability = float(probabilities[predicted_index])
        probability_map = {
            name: round(float(probabilities[index]), 6)
            for index, name in enumerate(self.target_names)
        }

        if prediction_probability < self.min_prediction_probability:
            state = "uncertain"
            reason = "prediction_probability_below_validation_threshold"
        else:
            state = self.target_names[predicted_index]
            reason = None

        return {
            **base_result,
            "state": state,
            "reason": reason,
            "prediction_probability": round(prediction_probability, 6),
            "probabilities": probability_map,
        }


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        return Path(model_path).expanduser().resolve()
    configured_path = os.environ.get("GAZE_SHIFT_MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_PATH


@lru_cache(maxsize=4)
def _load_classifier_cached(model_path: str) -> GazeShiftClassifier:
    path = Path(model_path)
    if not path.is_file():
        raise GazeShiftModelUnavailable(
            "Binary gaze-shift model not found. Run "
            "`python experiments/vision_events_v2/train_baselines.py` first, "
            "or set GAZE_SHIFT_MODEL_PATH."
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise GazeShiftModelUnavailable(
            f"Could not load binary gaze-shift model {path.name}: {exc}"
        ) from exc
    return GazeShiftClassifier(payload, model_name=path.name)


def load_gaze_shift_classifier(
    model_path: str | Path | None = None,
) -> GazeShiftClassifier:
    return _load_classifier_cached(str(resolve_model_path(model_path)))


def _read_window_sequence(
    capture: cv2.VideoCapture,
    face_mesh,
    start: float,
    end: float,
    sequence_length: int,
    fps: float,
) -> np.ndarray:
    sample_times = np.linspace(start, end, sequence_length, endpoint=False)
    target_frames = np.rint(sample_times * fps).astype(int)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_frames[0]))
    current_frame_index = int(target_frames[0])
    last_frame = None
    vectors = []

    for target_frame in target_frames:
        while current_frame_index <= target_frame:
            ok, frame = capture.read()
            if not ok:
                last_frame = None
                break
            last_frame = frame
            current_frame_index += 1
        vectors.append(extract_frame_features(last_frame, face_mesh))

    return np.stack(vectors).astype(np.float32)


def _incomplete_window(start: float, end: float) -> dict:
    return {
        "window_start": round(start, 2),
        "window_end": round(end, 2),
        "state": "unobservable",
        "reason": "incomplete_window",
        "face_visibility_ratio": None,
        "visual_observability_ratio": None,
        "prediction_probability": None,
        "probabilities": {
            "no_gaze_shift": None,
            "gaze_shift": None,
        },
    }


def classify_gaze_shift_video(
    video_path: str | Path,
    model_path: str | Path | None = None,
) -> dict:
    """Classify each non-overlapping three-second window of an uploaded video."""
    classifier = load_gaze_shift_classifier(model_path)
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for gaze-shift inference: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Video FPS or frame count is unavailable for gaze-shift inference.")
    duration = frame_count / fps

    windows = []
    face_mesh_solution = mp.solutions.face_mesh
    with face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        window_start = 0.0
        while window_start < duration:
            window_end = min(window_start + classifier.window_seconds, duration)
            if window_end - window_start < classifier.window_seconds - (1.0 / fps):
                windows.append(_incomplete_window(window_start, window_end))
                break

            sequence = _read_window_sequence(
                capture,
                face_mesh,
                window_start,
                window_start + classifier.window_seconds,
                classifier.sequence_length,
                fps,
            )
            result = classifier.predict_sequence(sequence)
            windows.append(
                {
                    "window_start": round(window_start, 2),
                    "window_end": round(window_start + classifier.window_seconds, 2),
                    **result,
                }
            )
            window_start += classifier.window_seconds
    capture.release()

    counts = Counter(window["state"] for window in windows)
    classified_windows = counts["no_gaze_shift"] + counts["gaze_shift"]
    observable_windows = classified_windows + counts["uncertain"]
    gaze_shift_ratio = (
        counts["gaze_shift"] / classified_windows if classified_windows else None
    )
    return {
        "task": "binary_gaze_shift",
        "target_names": list(TARGET_NAMES),
        "model": classifier.model_name,
        "decision_policy": classifier.decision_policy,
        "summary": {
            "total_windows": len(windows),
            "observable_windows": observable_windows,
            "classified_windows": classified_windows,
            "no_gaze_shift_windows": counts["no_gaze_shift"],
            "gaze_shift_windows": counts["gaze_shift"],
            "uncertain_windows": counts["uncertain"],
            "unobservable_windows": counts["unobservable"],
            "gaze_shift_ratio": (
                round(gaze_shift_ratio, 4) if gaze_shift_ratio is not None else None
            ),
        },
        "windows": windows,
    }
