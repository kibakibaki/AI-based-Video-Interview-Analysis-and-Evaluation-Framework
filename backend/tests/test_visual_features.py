import unittest

from Vision.face_analysis.analysis_utils import (
    AttentionReferenceTracker,
    _fixed_primary_attention_states,
)
from Vision.face_analysis.visual_features import VisualFrameObservation
from Vision.face_analysis.visual_features import VisualFeatureTracker


class VisualFeatureTrackerTest(unittest.TestCase):
    def test_full_video_reference_is_applied_to_each_window_observation(self):
        reference_tracker = AttentionReferenceTracker(min_samples=30)
        for _ in range(5):
            reference_tracker.update_and_check(
                pitch=0.0,
                yaw=25.0,
                yaw_threshold=25.0,
                pitch_threshold=20.0,
                gaze_observation=None,
            )

        reference_horizontal, reference_vertical = (
            reference_tracker.dominant_reference_offset()
        )
        observations = [
            VisualFrameObservation(0.0, True, False, False, pitch=0.0, yaw=25.0),
            VisualFrameObservation(1.0, True, False, False, pitch=0.0, yaw=75.0),
        ]

        states = _fixed_primary_attention_states(
            observations,
            reference_horizontal,
            reference_vertical,
            yaw_threshold=25.0,
            pitch_threshold=20.0,
        )

        self.assertEqual(reference_horizontal, 1.0)
        self.assertEqual(reference_vertical, 0.0)
        self.assertEqual(states, [True, False])

    def test_fixed_primary_attention_drives_window_look_away_features(self):
        tracker = VisualFeatureTracker()

        for second, camera_contact in enumerate([True, False, False, False]):
            tracker.update(
                current_time=float(second),
                face_detected=True,
                looking_at_camera=camera_contact,
                looking_at_primary=camera_contact,
            )

        tracker.set_primary_attention_states(
            [True, False, False, True],
            reference_horizontal=0.5,
            reference_vertical=-0.5,
        )

        summary = tracker.finish(total_duration=4.0)
        window = tracker.window_features(
            total_duration=4.0,
            window_size=4.0,
            step_size=4.0,
        )[0]

        self.assertEqual(summary["eye_contact_ratio"], 0.25)
        self.assertEqual(summary["primary_attention_ratio"], 0.5)
        self.assertEqual(summary["look_away_ratio"], 0.5)
        self.assertEqual(summary["looking_away_total_time"], 2.0)
        self.assertEqual(window["primary_attention_ratio"], 0.5)
        self.assertEqual(window["look_away_ratio"], 0.5)
        self.assertEqual(window["looking_away_total_time"], 2.0)
        self.assertEqual(window["primary_attention_reference_horizontal"], 0.5)
        self.assertEqual(window["primary_attention_reference_vertical"], -0.5)

    def test_missing_face_is_not_counted_as_look_away(self):
        tracker = VisualFeatureTracker()
        tracker.update(0.0, True, True, looking_at_primary=True)
        tracker.update(1.0, False, False, looking_at_primary=False)
        tracker.update(2.0, True, False, looking_at_primary=False)

        tracker.set_primary_attention_states([True, False, False], 0.0, 0.0)
        window = tracker.window_features(
            total_duration=3.0,
            window_size=3.0,
            step_size=3.0,
        )[0]

        self.assertEqual(window["primary_attention_ratio"], 0.5)
        self.assertEqual(window["look_away_ratio"], 0.5)
        self.assertEqual(window["looking_away_total_time"], 1.0)


if __name__ == "__main__":
    unittest.main()
