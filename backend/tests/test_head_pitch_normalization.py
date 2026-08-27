import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Vision.face_analysis.head_pose_utils import (  # noqa: E402
    angle_distance_from_front,
    signed_angle_from_front,
)
from Vision.face_analysis import gaze_shift_features  # noqa: E402


class SignedAngleFromFrontTests(unittest.TestCase):
    def test_maps_both_front_facing_euler_representations_to_signed_offsets(self):
        cases = {
            0.0: 0.0,
            10.0: 10.0,
            -10.0: -10.0,
            170.0: -10.0,
            -170.0: 10.0,
            179.0: -1.0,
            -179.0: 1.0,
            180.0: 0.0,
            -180.0: 0.0,
        }

        for raw_angle, expected_offset in cases.items():
            with self.subTest(raw_angle=raw_angle):
                self.assertEqual(signed_angle_from_front(raw_angle), expected_offset)
                self.assertEqual(
                    angle_distance_from_front(raw_angle),
                    abs(expected_offset),
                )

    def test_stays_continuous_across_positive_to_negative_180_wrap(self):
        raw_angles = [170.0, 175.0, 179.0, 180.0, -179.0, -175.0, -170.0]

        offsets = [signed_angle_from_front(angle) for angle in raw_angles]

        self.assertEqual(offsets, [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        self.assertLessEqual(max(abs(np.diff(offsets))), 5.0)

    def test_stays_continuous_across_negative_to_positive_180_wrap(self):
        raw_angles = [-170.0, -175.0, -179.0, -180.0, 179.0, 175.0, 170.0]

        offsets = [signed_angle_from_front(angle) for angle in raw_angles]

        self.assertEqual(offsets, [10.0, 5.0, 1.0, 0.0, -1.0, -5.0, -10.0])
        self.assertLessEqual(max(abs(np.diff(offsets))), 5.0)


class V2HeadPitchFeatureTests(unittest.TestCase):
    @staticmethod
    def _face_mesh_with_landmarks():
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]

        coordinates = {
            13: (0.50, 0.68),
            14: (0.50, 0.72),
            33: (0.20, 0.40),
            78: (0.35, 0.70),
            105: (0.30, 0.30),
            133: (0.40, 0.40),
            144: (0.25, 0.42),
            145: (0.30, 0.42),
            153: (0.35, 0.42),
            158: (0.35, 0.38),
            159: (0.30, 0.38),
            160: (0.25, 0.38),
            263: (0.80, 0.40),
            308: (0.65, 0.70),
            334: (0.70, 0.30),
            362: (0.60, 0.40),
            373: (0.75, 0.42),
            374: (0.70, 0.42),
            380: (0.65, 0.42),
            385: (0.65, 0.38),
            386: (0.70, 0.38),
            387: (0.75, 0.38),
        }
        for index, (x, y) in coordinates.items():
            points[index] = SimpleNamespace(x=x, y=y)

        for index in range(468, 473):
            points[index] = SimpleNamespace(x=0.30, y=0.40)
        for index in range(473, 478):
            points[index] = SimpleNamespace(x=0.70, y=0.40)

        landmarks = SimpleNamespace(landmark=points)
        result = SimpleNamespace(multi_face_landmarks=[landmarks])
        return SimpleNamespace(process=lambda _rgb: result)

    def test_v2_feature_uses_signed_front_pitch_instead_of_raw_euler_pitch(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pitch_index = gaze_shift_features.FEATURE_NAMES.index("head_pitch")

        for raw_pitch, expected_pitch in ((179.0, -1.0), (-179.0, 1.0)):
            with self.subTest(raw_pitch=raw_pitch):
                with patch.object(
                    gaze_shift_features,
                    "estimate_head_pose",
                    return_value=(raw_pitch, 5.0, 6.0),
                ):
                    features = gaze_shift_features.extract_frame_features(
                        frame,
                        self._face_mesh_with_landmarks(),
                    )

                self.assertEqual(features[pitch_index], expected_pitch)


if __name__ == "__main__":
    unittest.main()
