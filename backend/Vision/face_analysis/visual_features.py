import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass
class VisualFrameObservation:
    time: float
    face_detected: bool
    looking_at_camera: bool
    looking_at_primary: bool
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
    looking_at_primary_frames: int = 0
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
    primary_attention_reference_horizontal: float | None = None
    primary_attention_reference_vertical: float | None = None
    primary_attention_window_references: dict[
        tuple[float, float],
        tuple[float | None, float | None],
    ] = field(default_factory=dict)

    def update(
        self,
        current_time,
        face_detected,
        looking_at_camera,
        looking_at_primary=None,
        pitch=None,
        yaw=None,
        roll=None,
        gaze_observation=None,
    ):
        observation = self._make_observation(
            current_time=current_time,
            face_detected=face_detected,
            looking_at_camera=looking_at_camera,
            looking_at_primary=(
                looking_at_camera if looking_at_primary is None else looking_at_primary
            ),
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

        if observation.looking_at_primary:
            self.looking_at_primary_frames += 1

        self._update_looking_away_segments(
            current_time,
            face_detected,
            observation.looking_at_primary,
        )
        self._update_head_pose(pitch, yaw, roll)
        self._update_gaze(gaze_observation)

    def finish(self, total_duration):
        self._rebuild_looking_away_segments(total_duration)

        attention_context = self._attention_context(self.observations)
        self._add_primary_attention_reference(attention_context)
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
            "primary_attention_ratio": self._safe_ratio(
                self.looking_at_primary_frames,
                self.face_detected_frames,
            ),
            "look_away_ratio": self._safe_ratio(
                self.face_detected_frames - self.looking_at_primary_frames,
                self.face_detected_frames,
            ),
            "primary_attention_reference_horizontal": (
                self.primary_attention_reference_horizontal
            ),
            "primary_attention_reference_vertical": (
                self.primary_attention_reference_vertical
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
            **self._attention_features_for_observations(self.observations, attention_context),
        }

    def window_features(self, total_duration, window_size=3.0, step_size=3.0):
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
                attention_context = self._attention_context(window_observations)
                self._add_primary_attention_reference(
                    attention_context,
                    window_start,
                    window_end,
                )
                rows.append(self._features_for_window(
                    window_start,
                    window_end,
                    window_observations,
                    attention_context,
                ))

            if window_end >= total_duration:
                break
            window_start += step_size

        return rows

    def set_primary_attention_states(
        self,
        states,
        reference_horizontal,
        reference_vertical,
    ):
        """Apply one shared attention reference, used for live camera analysis."""
        if len(states) != len(self.observations):
            raise ValueError("Primary-attention states must match the observation count")

        for observation, state in zip(self.observations, states):
            observation.looking_at_primary = bool(state) and observation.face_detected

        self.looking_at_primary_frames = sum(
            1 for observation in self.observations if observation.looking_at_primary
        )
        self.primary_attention_reference_horizontal = reference_horizontal
        self.primary_attention_reference_vertical = reference_vertical
        self.primary_attention_window_references = {}

    def set_window_primary_attention_states(self, states, references):
        """Apply independently calculated primary-attention states per window."""
        if len(states) != len(self.observations):
            raise ValueError("Primary-attention states must match the observation count")

        for observation, state in zip(self.observations, states):
            observation.looking_at_primary = bool(state) and observation.face_detected

        self.looking_at_primary_frames = sum(
            1 for observation in self.observations if observation.looking_at_primary
        )
        self.primary_attention_reference_horizontal = None
        self.primary_attention_reference_vertical = None
        self.primary_attention_window_references = dict(references)

    def _add_primary_attention_reference(
        self,
        attention_context,
        window_start=None,
        window_end=None,
    ):
        if window_start is not None and window_end is not None:
            reference = self.primary_attention_window_references.get(
                (round(window_start, 2), round(window_end, 2))
            )
            if reference is not None:
                reference_horizontal, reference_vertical = reference
                attention_context["primary_attention_reference_horizontal"] = (
                    reference_horizontal
                )
                attention_context["primary_attention_reference_vertical"] = (
                    reference_vertical
                )
                return

        attention_context["primary_attention_reference_horizontal"] = (
            self.primary_attention_reference_horizontal
        )
        attention_context["primary_attention_reference_vertical"] = (
            self.primary_attention_reference_vertical
        )

    @staticmethod
    def _make_observation(
        current_time,
        face_detected,
        looking_at_camera,
        looking_at_primary,
        pitch=None,
        yaw=None,
        roll=None,
        gaze_observation=None,
    ):
        return VisualFrameObservation(
            time=current_time,
            face_detected=face_detected,
            looking_at_camera=looking_at_camera,
            looking_at_primary=looking_at_primary,
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
    def _features_for_window(cls, window_start, window_end, observations, attention_context):
        total_frames = len(observations)
        face_detected_frames = sum(1 for obs in observations if obs.face_detected)
        looking_at_camera_frames = sum(1 for obs in observations if obs.looking_at_camera)
        looking_at_primary_frames = sum(1 for obs in observations if obs.looking_at_primary)
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
            "primary_attention_ratio": cls._safe_ratio(
                looking_at_primary_frames,
                face_detected_frames,
            ),
            "look_away_ratio": cls._safe_ratio(
                face_detected_frames - looking_at_primary_frames,
                face_detected_frames,
            ),
            "primary_attention_reference_horizontal": (
                attention_context.get("primary_attention_reference_horizontal")
            ),
            "primary_attention_reference_vertical": (
                attention_context.get("primary_attention_reference_vertical")
            ),
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
            **cls._attention_features_for_observations(observations, attention_context),
        }

    @classmethod
    def _attention_context(cls, observations):
        gaze_points = cls._gaze_points(observations)
        head_points = cls._head_points(observations)

        context = {
            "primary_gaze_horizontal_ratio": None,
            "primary_gaze_vertical_ratio": None,
            "primary_gaze_zone_ratio": 0.0,
            "secondary_gaze_zone_ratio": 0.0,
            "primary_head_pitch": None,
            "primary_head_yaw": None,
            "primary_attention_reference_horizontal": None,
            "primary_attention_reference_vertical": None,
        }

        if gaze_points:
            primary_bin, primary_count, secondary_count = cls._dominant_bin_counts(gaze_points, bin_size=0.12)
            primary_points = [
                point
                for point in gaze_points
                if cls._bin_point(point, 0.12) == primary_bin
            ]
            context["primary_gaze_horizontal_ratio"] = cls._rounded_mean([point[0] for point in primary_points])
            context["primary_gaze_vertical_ratio"] = cls._rounded_mean([point[1] for point in primary_points])
            context["primary_gaze_zone_ratio"] = cls._safe_ratio(primary_count, len(gaze_points))
            context["secondary_gaze_zone_ratio"] = cls._safe_ratio(secondary_count, len(gaze_points))

        if head_points:
            primary_bin, _, _ = cls._dominant_bin_counts(head_points, bin_size=8.0)
            primary_points = [
                point
                for point in head_points
                if cls._bin_point(point, 8.0) == primary_bin
            ]
            context["primary_head_pitch"] = cls._rounded_mean([point[0] for point in primary_points])
            context["primary_head_yaw"] = cls._rounded_mean([point[1] for point in primary_points])

        return context

    @classmethod
    def _attention_features_for_observations(cls, observations, attention_context):
        gaze_points = cls._gaze_points(observations)
        head_points = cls._head_points(observations)

        primary_gaze = (
            attention_context.get("primary_gaze_horizontal_ratio"),
            attention_context.get("primary_gaze_vertical_ratio"),
        )
        gaze_distances = cls._distances_from_primary(gaze_points, primary_gaze)

        primary_head = (
            attention_context.get("primary_head_pitch"),
            attention_context.get("primary_head_yaw"),
        )
        head_distances = cls._distances_from_primary(head_points, primary_head)

        return {
            "primary_gaze_horizontal_ratio": attention_context.get("primary_gaze_horizontal_ratio"),
            "primary_gaze_vertical_ratio": attention_context.get("primary_gaze_vertical_ratio"),
            "primary_gaze_zone_ratio": attention_context.get("primary_gaze_zone_ratio", 0.0),
            "secondary_gaze_zone_ratio": attention_context.get("secondary_gaze_zone_ratio", 0.0),
            "gaze_deviation_from_primary_ratio": cls._ratio_over_threshold(gaze_distances, 0.18),
            "gaze_deviation_from_primary_mean": cls._rounded_mean(gaze_distances),
            "primary_head_pitch": attention_context.get("primary_head_pitch"),
            "primary_head_yaw": attention_context.get("primary_head_yaw"),
            "head_deviation_from_primary_ratio": cls._ratio_over_threshold(head_distances, 12.0),
            "head_deviation_from_primary_mean": cls._rounded_mean(head_distances),
        }

    @staticmethod
    def _gaze_points(observations):
        return [
            (obs.horizontal_ratio, obs.vertical_ratio)
            for obs in observations
            if obs.horizontal_ratio is not None and obs.vertical_ratio is not None
        ]

    @staticmethod
    def _head_points(observations):
        return [
            (obs.pitch, obs.yaw)
            for obs in observations
            if obs.pitch is not None and obs.yaw is not None
        ]

    @staticmethod
    def _bin_point(point, bin_size):
        return tuple(round(value / bin_size) for value in point)

    @classmethod
    def _dominant_bin_counts(cls, points, bin_size):
        counts = Counter(cls._bin_point(point, bin_size) for point in points)
        most_common = counts.most_common(2)
        primary_bin, primary_count = most_common[0]
        secondary_count = most_common[1][1] if len(most_common) > 1 else 0
        return primary_bin, primary_count, secondary_count

    @staticmethod
    def _distances_from_primary(points, primary_point):
        if not points or primary_point[0] is None or primary_point[1] is None:
            return []

        primary_x, primary_y = primary_point
        return [
            math.sqrt((point[0] - primary_x) ** 2 + (point[1] - primary_y) ** 2)
            for point in points
        ]

    @classmethod
    def _ratio_over_threshold(cls, values, threshold):
        if not values:
            return 0.0
        return cls._safe_ratio(sum(1 for value in values if value > threshold), len(values))

    def _update_looking_away_segments(
        self,
        current_time,
        face_detected,
        looking_at_primary,
    ):
        if not face_detected or looking_at_primary:
            if self.current_looking_away_start is not None:
                self.looking_away_segments.append((self.current_looking_away_start, current_time))
                self.current_looking_away_start = None
            return

        if self.current_looking_away_start is None:
            self.current_looking_away_start = current_time

    def _rebuild_looking_away_segments(self, total_duration):
        self.current_looking_away_start = None
        self.looking_away_segments = []

        for observation in self.observations:
            self._update_looking_away_segments(
                observation.time,
                observation.face_detected,
                observation.looking_at_primary,
            )

        if self.current_looking_away_start is not None:
            self.looking_away_segments.append(
                (self.current_looking_away_start, total_duration)
            )
            self.current_looking_away_start = None

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
            if not obs.face_detected or obs.looking_at_primary:
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
