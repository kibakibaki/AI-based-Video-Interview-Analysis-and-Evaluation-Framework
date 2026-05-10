from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass
class VisualFrameObservation:
    time: float
    face_detected: bool
    looking_at_camera: bool
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    gaze_direction: str | None = None
    gaze_is_center: bool = False
    is_blinking: bool = False
    horizontal_ratio: float | None = None
    vertical_ratio: float | None = None


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
    observations: list[VisualFrameObservation] = field(default_factory=list)

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
        observation = self._make_observation(
            current_time=current_time,
            face_detected=face_detected,
            looking_at_camera=looking_at_camera,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_observation=gaze_observation,
        )
        self.observations.append(observation)

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
            "head_pitch_offset_mean": self._rounded_mean(self._pitch_offsets(self.pitch_values)),
            "head_pitch_offset_std": self._rounded_std(self._pitch_offsets(self.pitch_values)),
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

    def window_features(self, total_duration, window_size=5.0, step_size=1.0):
        if total_duration <= 0 or not self.observations:
            return []

        rows = []
        window_start = 0.0
        while window_start < total_duration:
            window_end = min(window_start + window_size, total_duration)
            if window_end <= window_start:
                break

            window_observations = [
                observation
                for observation in self.observations
                if window_start <= observation.time < window_end
            ]
            if window_observations:
                rows.append(self._features_for_window(
                    window_start,
                    window_end,
                    window_observations,
                ))

            if window_end >= total_duration:
                break
            window_start += step_size

        return rows

    @staticmethod
    def _make_observation(
        current_time,
        face_detected,
        looking_at_camera,
        pitch=None,
        yaw=None,
        roll=None,
        gaze_observation=None,
    ):
        return VisualFrameObservation(
            time=current_time,
            face_detected=face_detected,
            looking_at_camera=looking_at_camera,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_direction=getattr(gaze_observation, "direction", None),
            gaze_is_center=bool(getattr(gaze_observation, "is_center", False)),
            is_blinking=bool(getattr(gaze_observation, "is_blinking", False)),
            horizontal_ratio=getattr(gaze_observation, "horizontal_ratio", None),
            vertical_ratio=getattr(gaze_observation, "vertical_ratio", None),
        )

    @classmethod
    def _features_for_window(cls, window_start, window_end, observations):
        total_frames = len(observations)
        face_detected_frames = sum(1 for obs in observations if obs.face_detected)
        looking_at_camera_frames = sum(1 for obs in observations if obs.looking_at_camera)
        valid_gaze_frames = sum(1 for obs in observations if obs.gaze_direction is not None)
        gaze_center_frames = sum(1 for obs in observations if obs.gaze_is_center)
        blink_count = cls._count_blink_events(observations)
        duration = window_end - window_start

        pitch_values = [obs.pitch for obs in observations if obs.pitch is not None]
        pitch_offsets = cls._pitch_offsets(pitch_values)
        yaw_values = [obs.yaw for obs in observations if obs.yaw is not None]
        roll_values = [obs.roll for obs in observations if obs.roll is not None]
        horizontal_ratios = [
            obs.horizontal_ratio
            for obs in observations
            if obs.horizontal_ratio is not None
        ]
        vertical_ratios = [
            obs.vertical_ratio
            for obs in observations
            if obs.vertical_ratio is not None
        ]
        head_delta_values = cls._head_delta_values(observations)
        looking_away_total_time = cls._looking_away_total_time(observations, window_start, window_end)

        return {
            "window_start": round(window_start, 2),
            "window_end": round(window_end, 2),
            "window_duration": round(duration, 2),
            "window_frames": total_frames,
            "eye_contact_ratio": cls._safe_ratio(looking_at_camera_frames, face_detected_frames),
            "face_visibility_ratio": cls._safe_ratio(face_detected_frames, total_frames),
            "gaze_center_ratio": cls._safe_ratio(gaze_center_frames, valid_gaze_frames),
            "valid_gaze_frames": valid_gaze_frames,
            "gaze_center_frames": gaze_center_frames,
            "looking_away_total_time": round(looking_away_total_time, 2),
            "head_pitch_offset_mean": cls._rounded_mean(pitch_offsets),
            "head_pitch_offset_std": cls._rounded_std(pitch_offsets),
            "head_yaw_mean": cls._rounded_mean(yaw_values),
            "head_yaw_std": cls._rounded_std(yaw_values),
            "head_roll_mean": cls._rounded_mean(roll_values),
            "head_roll_std": cls._rounded_std(roll_values),
            "head_movement_mean_delta": cls._rounded_mean(head_delta_values),
            "head_movement_stability_score": cls._head_movement_stability_score_from_values(head_delta_values),
            "blink_count": blink_count,
            "blink_rate_per_minute": cls._rate_per_minute(blink_count, duration),
            "gaze_horizontal_ratio_mean": cls._rounded_mean(horizontal_ratios),
            "gaze_horizontal_ratio_std": cls._rounded_std(horizontal_ratios),
            "gaze_vertical_ratio_mean": cls._rounded_mean(vertical_ratios),
            "gaze_vertical_ratio_std": cls._rounded_std(vertical_ratios),
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
        return self._head_movement_stability_score_from_values(self.head_delta_values)

    @classmethod
    def _head_movement_stability_score_from_values(cls, values):
        mean_delta = cls._safe_mean(values)
        if mean_delta is None:
            return None

        return round(100 / (1 + mean_delta), 2)

    @staticmethod
    def _head_delta_values(observations):
        values = []
        previous_pose = None

        for obs in observations:
            if obs.pitch is None or obs.yaw is None or obs.roll is None:
                continue

            current_pose = (obs.pitch, obs.yaw, obs.roll)
            if previous_pose is not None:
                previous_pitch, previous_yaw, previous_roll = previous_pose
                delta = (
                    abs(obs.pitch - previous_pitch)
                    + abs(obs.yaw - previous_yaw)
                    + abs(obs.roll - previous_roll)
                ) / 3
                values.append(delta)

            previous_pose = current_pose

        return values

    @staticmethod
    def _count_blink_events(observations):
        count = 0
        previous_blinking = False

        for obs in observations:
            if obs.is_blinking and not previous_blinking:
                count += 1
            previous_blinking = obs.is_blinking

        return count

    @staticmethod
    def _looking_away_total_time(observations, window_start, window_end):
        if not observations:
            return 0.0

        duration = 0.0
        for index, obs in enumerate(observations):
            if obs.looking_at_camera:
                continue

            if index + 1 < len(observations):
                next_time = observations[index + 1].time
            else:
                next_time = window_end

            duration += max(0.0, min(next_time, window_end) - max(obs.time, window_start))

        return duration

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
    def _pitch_offset(pitch):
        return min(abs(pitch), abs(abs(pitch) - 180))

    @classmethod
    def _pitch_offsets(cls, values):
        return [cls._pitch_offset(value) for value in values]

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
        return self._rate_per_minute(self.blink_count, total_duration)

    @staticmethod
    def _rate_per_minute(count, duration):
        if duration <= 0:
            return 0.0
        return round((count / duration) * 60, 4)
