from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass
class VisualFeatureTracker:
    total_frames: int = 0
    face_detected_frames: int = 0
    looking_at_camera_frames: int = 0
    valid_gaze_frames: int = 0
    gaze_center_frames: int = 0
    blink_count: int = 0
    previous_blinking: bool = False
    current_looking_away_start: float | None = None
    looking_away_segments: list[tuple[float, float]] = field(default_factory=list)
    pitch_values: list[float] = field(default_factory=list)
    yaw_values: list[float] = field(default_factory=list)
    roll_values: list[float] = field(default_factory=list)
    head_delta_values: list[float] = field(default_factory=list)
    horizontal_gaze_ratios: list[float] = field(default_factory=list)
    vertical_gaze_ratios: list[float] = field(default_factory=list)
    previous_head_pose: tuple[float, float, float] | None = None

    def update(
        self,
        current_time,
        face_detected,
        looking_at_camera,
        pitch=None,
        yaw=None,
        roll=None,
        gaze_observation=None,
    ):
        self.total_frames += 1

        if face_detected:
            self.face_detected_frames += 1

        if looking_at_camera:
            self.looking_at_camera_frames += 1

        self._update_looking_away_segments(current_time, looking_at_camera)
        self._update_head_pose(pitch, yaw, roll)
        self._update_gaze(gaze_observation)

    def finish(self, total_duration):
        if self.current_looking_away_start is not None:
            self.looking_away_segments.append((self.current_looking_away_start, total_duration))
            self.current_looking_away_start = None

        looking_away_durations = [
            end - start
            for start, end in self.looking_away_segments
        ]
        looking_away_total_time = sum(looking_away_durations)
        longest_looking_away_duration = max(looking_away_durations, default=0.0)

        return {
            "eye_contact_ratio": self._safe_ratio(
                self.looking_at_camera_frames,
                self.face_detected_frames,
            ),
            "face_visibility_ratio": self._safe_ratio(
                self.face_detected_frames,
                self.total_frames,
            ),
            "gaze_center_ratio": self._safe_ratio(
                self.gaze_center_frames,
                self.valid_gaze_frames,
            ),
            "valid_gaze_frames": self.valid_gaze_frames,
            "gaze_center_frames": self.gaze_center_frames,
            "looking_away_total_time": round(looking_away_total_time, 2),
            "longest_looking_away_duration": round(longest_looking_away_duration, 2),
            "looking_away_segments": self._serialise_segments(self.looking_away_segments),
            "head_pitch_mean": self._rounded_mean(self.pitch_values),
            "head_pitch_std": self._rounded_std(self.pitch_values),
            "head_yaw_mean": self._rounded_mean(self.yaw_values),
            "head_yaw_std": self._rounded_std(self.yaw_values),
            "head_roll_mean": self._rounded_mean(self.roll_values),
            "head_roll_std": self._rounded_std(self.roll_values),
            "head_movement_mean_delta": self._rounded_mean(self.head_delta_values),
            "head_movement_stability_score": self._head_movement_stability_score(),
            "blink_count": self.blink_count,
            "blink_rate_per_minute": self._blink_rate_per_minute(total_duration),
            "gaze_horizontal_ratio_mean": self._rounded_mean(self.horizontal_gaze_ratios),
            "gaze_horizontal_ratio_std": self._rounded_std(self.horizontal_gaze_ratios),
            "gaze_vertical_ratio_mean": self._rounded_mean(self.vertical_gaze_ratios),
            "gaze_vertical_ratio_std": self._rounded_std(self.vertical_gaze_ratios),
        }

    def _update_looking_away_segments(self, current_time, looking_at_camera):
        if looking_at_camera:
            if self.current_looking_away_start is not None:
                self.looking_away_segments.append((self.current_looking_away_start, current_time))
                self.current_looking_away_start = None
            return

        if self.current_looking_away_start is None:
            self.current_looking_away_start = current_time

    def _update_head_pose(self, pitch, yaw, roll):
        if pitch is None or yaw is None or roll is None:
            return

        current_pose = (pitch, yaw, roll)
        self.pitch_values.append(pitch)
        self.yaw_values.append(yaw)
        self.roll_values.append(roll)

        if self.previous_head_pose is not None:
            previous_pitch, previous_yaw, previous_roll = self.previous_head_pose
            delta = (
                abs(pitch - previous_pitch)
                + abs(yaw - previous_yaw)
                + abs(roll - previous_roll)
            ) / 3
            self.head_delta_values.append(delta)

        self.previous_head_pose = current_pose

    def _update_gaze(self, gaze_observation):
        if gaze_observation is None:
            self.previous_blinking = False
            return

        if gaze_observation.direction is not None:
            self.valid_gaze_frames += 1

        if gaze_observation.is_center:
            self.gaze_center_frames += 1

        if gaze_observation.is_blinking and not self.previous_blinking:
            self.blink_count += 1
        self.previous_blinking = gaze_observation.is_blinking

        if gaze_observation.horizontal_ratio is not None:
            self.horizontal_gaze_ratios.append(gaze_observation.horizontal_ratio)
        if gaze_observation.vertical_ratio is not None:
            self.vertical_gaze_ratios.append(gaze_observation.vertical_ratio)

    def _head_movement_stability_score(self):
        mean_delta = self._safe_mean(self.head_delta_values)
        if mean_delta is None:
            return None

        return round(100 / (1 + mean_delta), 2)

    @staticmethod
    def _serialise_segments(segments):
        return [
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
            }
            for start, end in segments
        ]

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    @staticmethod
    def _safe_mean(values):
        if not values:
            return None
        return mean(values)

    @classmethod
    def _rounded_mean(cls, values):
        value = cls._safe_mean(values)
        if value is None:
            return None
        return round(value, 4)

    @staticmethod
    def _rounded_std(values):
        if len(values) < 2:
            return None
        return round(pstdev(values), 4)

    def _blink_rate_per_minute(self, total_duration):
        if total_duration <= 0:
            return 0.0
        return round((self.blink_count / total_duration) * 60, 4)
