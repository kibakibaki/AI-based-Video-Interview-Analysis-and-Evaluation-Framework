import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Vision.face_analysis.gaze_shift_classifier import (  # noqa: E402
    GazeShiftClassifier,
    GazeShiftModelContractError,
    summarise_sequence,
)
from Vision.face_analysis.gaze_shift_features import FEATURE_NAMES  # noqa: E402


class FakeBinaryModel:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return np.tile(self.probabilities, (len(X), 1))


def make_payload(probabilities=(0.2, 0.8), threshold=0.65):
    return {
        "model": FakeBinaryModel(probabilities),
        "source_feature_names": list(FEATURE_NAMES),
        "metadata": {
            "target": "binary",
            "target_names": ["no_gaze_shift", "gaze_shift"],
            "sample_rate_hz": 10.0,
            "window_seconds": 3.0,
            "sequence_length": 30,
        },
        "decision_policy": {
            "min_face_visibility": 0.5,
            "min_visual_observability": 0.5,
            "min_prediction_probability": threshold,
            "threshold_source": "validation_test_fixture",
        },
    }


def visible_sequence(face_visibility=1.0):
    sequence = np.ones((30, len(FEATURE_NAMES)), dtype=np.float32)
    sequence[:, 0] = face_visibility
    return sequence


class GazeShiftDecisionTests(unittest.TestCase):
    def test_low_face_visibility_returns_unobservable_without_model_call(self):
        payload = make_payload()
        classifier = GazeShiftClassifier(payload)

        result = classifier.predict_sequence(visible_sequence(0.4))

        self.assertEqual(result["state"], "unobservable")
        self.assertEqual(payload["model"].calls, 0)

    def test_low_probability_returns_uncertain(self):
        classifier = GazeShiftClassifier(make_payload((0.45, 0.55), threshold=0.6))

        result = classifier.predict_sequence(visible_sequence())

        self.assertEqual(result["state"], "uncertain")
        self.assertEqual(
            result["reason"],
            "prediction_probability_below_validation_threshold",
        )

    def test_missing_eye_and_pose_features_returns_unobservable(self):
        payload = make_payload()
        classifier = GazeShiftClassifier(payload)
        sequence = visible_sequence()
        iris_index = FEATURE_NAMES.index("iris_horizontal_mean")
        sequence[:16, iris_index] = np.nan

        result = classifier.predict_sequence(sequence)

        self.assertEqual(result["state"], "unobservable")
        self.assertEqual(
            result["reason"],
            "insufficient_eye_or_pose_observability",
        )
        self.assertEqual(payload["model"].calls, 0)

    def test_high_probability_returns_binary_gaze_state(self):
        classifier = GazeShiftClassifier(make_payload((0.2, 0.8), threshold=0.65))

        result = classifier.predict_sequence(visible_sequence())

        self.assertEqual(result["state"], "gaze_shift")
        self.assertEqual(result["probabilities"]["gaze_shift"], 0.8)

    def test_rejects_non_binary_model_contract(self):
        payload = make_payload()
        payload["metadata"]["target"] = "level"

        with self.assertRaises(GazeShiftModelContractError):
            GazeShiftClassifier(payload)

    def test_summary_matches_training_shape(self):
        summary = summarise_sequence(visible_sequence())

        self.assertEqual(summary.shape, (4 * len(FEATURE_NAMES),))


if __name__ == "__main__":
    unittest.main()
