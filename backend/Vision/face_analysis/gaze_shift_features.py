"""Shared frame-level features for the binary gaze-shift model."""

from __future__ import annotations

import cv2
import numpy as np

from .head_pose_utils import estimate_head_pose, signed_angle_from_front


FEATURE_NAMES = (
    "face_detected",
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "left_brow_eye_distance",
    "right_brow_eye_distance",
    "mouth_open_ratio",
    "mouth_width_ratio",
    "mouth_corner_asymmetry",
    "iris_horizontal_mean",
    "iris_vertical_mean",
    "head_pitch",
    "head_yaw",
    "head_roll",
)

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = tuple(range(468, 473))
RIGHT_IRIS = tuple(range(473, 478))


def _point(face_landmarks, index: int, width: int, height: int) -> np.ndarray:
    landmark = face_landmarks.landmark[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(points: list[np.ndarray]) -> float:
    horizontal = _distance(points[0], points[3])
    if horizontal <= 1e-6:
        return np.nan
    vertical = _distance(points[1], points[5]) + _distance(points[2], points[4])
    return vertical / (2.0 * horizontal)


def _projection_ratio(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    axis = end - start
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-6:
        return np.nan
    return float(np.dot(point - start, axis) / denominator)


def _iris_center(face_landmarks, indices: tuple[int, ...], width: int, height: int) -> np.ndarray:
    return np.mean(
        [_point(face_landmarks, index, width, height) for index in indices],
        axis=0,
    )


def missing_feature_vector() -> np.ndarray:
    """Retain face absence without inventing geometry."""
    return np.array([0.0, *([np.nan] * (len(FEATURE_NAMES) - 1))], dtype=np.float32)


def extract_landmark_features(frame: np.ndarray, face_landmarks) -> np.ndarray:
    """Extract one observable feature vector from already-detected landmarks."""
    if frame is None or frame.size == 0 or face_landmarks is None:
        return missing_feature_vector()

    height, width = frame.shape[:2]
    face_scale = _distance(
        _point(face_landmarks, 33, width, height),
        _point(face_landmarks, 263, width, height),
    )
    if face_scale <= 1e-6:
        return missing_feature_vector()

    left_eye_points = [_point(face_landmarks, index, width, height) for index in LEFT_EYE]
    right_eye_points = [_point(face_landmarks, index, width, height) for index in RIGHT_EYE]

    left_iris = _iris_center(face_landmarks, LEFT_IRIS, width, height)
    right_iris = _iris_center(face_landmarks, RIGHT_IRIS, width, height)
    left_horizontal = _projection_ratio(left_iris, left_eye_points[0], left_eye_points[3])
    right_horizontal = _projection_ratio(right_iris, right_eye_points[0], right_eye_points[3])
    left_vertical = _projection_ratio(
        left_iris,
        _point(face_landmarks, 159, width, height),
        _point(face_landmarks, 145, width, height),
    )
    right_vertical = _projection_ratio(
        right_iris,
        _point(face_landmarks, 386, width, height),
        _point(face_landmarks, 374, width, height),
    )

    mouth_left = _point(face_landmarks, 78, width, height)
    mouth_right = _point(face_landmarks, 308, width, height)
    mouth_width = _distance(mouth_left, mouth_right)
    mouth_open = _distance(
        _point(face_landmarks, 13, width, height),
        _point(face_landmarks, 14, width, height),
    )

    pose = estimate_head_pose(frame, face_landmarks)
    pitch, yaw, roll = (pose[:3] if pose is not None else (np.nan, np.nan, np.nan))
    pitch = signed_angle_from_front(pitch)

    return np.array(
        [
            1.0,
            _eye_aspect_ratio(left_eye_points),
            _eye_aspect_ratio(right_eye_points),
            _distance(
                _point(face_landmarks, 105, width, height),
                _point(face_landmarks, 159, width, height),
            ) / face_scale,
            _distance(
                _point(face_landmarks, 334, width, height),
                _point(face_landmarks, 386, width, height),
            ) / face_scale,
            mouth_open / max(mouth_width, 1e-6),
            mouth_width / face_scale,
            abs(float(mouth_left[1] - mouth_right[1])) / face_scale,
            float(np.nanmean([left_horizontal, right_horizontal])),
            float(np.nanmean([left_vertical, right_vertical])),
            float(pitch),
            float(yaw),
            float(roll),
        ],
        dtype=np.float32,
    )


def extract_frame_features(frame: np.ndarray, face_mesh) -> np.ndarray:
    """Extract one feature vector, including face detection, from a BGR frame."""
    if frame is None or frame.size == 0:
        return missing_feature_vector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return missing_feature_vector()
    return extract_landmark_features(frame, result.multi_face_landmarks[0])
