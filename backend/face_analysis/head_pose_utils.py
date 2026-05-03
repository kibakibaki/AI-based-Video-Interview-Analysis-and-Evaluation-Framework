import math

import cv2
import numpy as np


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees([pitch, yaw, roll])


def angle_distance_from_front(angle):
    """
    Return the smallest distance from a front-facing angle.
    solvePnP can represent a front-facing head as either 0 or +/-180 degrees.
    """
    return min(abs(angle), abs(abs(angle) - 180))


def is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold):
    pitch_offset = angle_distance_from_front(pitch)
    yaw_offset = abs(yaw)
    return yaw_offset <= yaw_threshold and pitch_offset <= pitch_threshold


def estimate_head_pose(frame, face_landmarks):
    """
    Estimate head pose using selected MediaPipe face landmarks.

    Returns:
        (pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs) or None
    """
    h, w, _ = frame.shape

    # Nose tip, chin, left eye outer, right eye outer, left mouth, right mouth.
    landmark_ids = [1, 152, 33, 263, 61, 291]

    image_points = []
    for idx in landmark_ids:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        image_points.append((x, y))

    image_points = np.array(image_points, dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye left corner
        (43.3, 32.7, -26.0),      # Right eye right corner
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1)      # Right mouth corner
    ])

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    return pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs


def draw_head_direction(frame, rvec, tvec, camera_matrix, dist_coeffs):
    nose_tip_3d = np.array([(0.0, 0.0, 0.0)])
    nose_end_3d = np.array([(0.0, 0.0, 100.0)])

    nose_tip_2d, _ = cv2.projectPoints(
        nose_tip_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    nose_end_2d, _ = cv2.projectPoints(
        nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs
    )

    start_point = tuple(nose_tip_2d[0][0].astype(int))
    end_point = tuple(nose_end_2d[0][0].astype(int))
    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
