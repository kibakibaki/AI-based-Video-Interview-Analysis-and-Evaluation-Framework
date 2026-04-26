# gaze_utils.py
import cv2
import os
import math
import mediapipe as mp
import numpy as np


def _rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])   # pitch
        y = math.atan2(-R[2, 0], sy)       # yaw
        z = math.atan2(R[1, 0], R[0, 0])   # roll
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def _get_head_pose(frame, face_landmarks):
    """
    Estimate head pose using selected face landmarks and solvePnP.
    Returns: (pitch, yaw, roll) or None
    """
    h, w, _ = frame.shape

    # MediaPipe Face Mesh landmark indices
    # nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
    landmark_ids = [1, 152, 33, 263, 61, 291]

    image_points = []
    for idx in landmark_ids:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        image_points.append((x, y))

    image_points = np.array(image_points, dtype="double")

    # Approximate 3D model points
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

    dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

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
    pitch, yaw, roll = _rotation_matrix_to_euler_angles(rotation_matrix)

    return pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs


def analyse_gaze_from_video(
    video_path,
    output_video_path=None,
    yaw_threshold=15,
    pitch_threshold=15,
    min_segment_duration=0.3
):
    """
    Analyse whether the person is roughly looking at the camera.

    Parameters:
        video_path: input video path
        output_video_path: optional annotated video output path
        yaw_threshold: abs(yaw) <= threshold => looking at camera
        pitch_threshold: abs(pitch) <= threshold => looking at camera
        min_segment_duration: minimum duration (seconds) to keep

    Returns:
        segments: list of (start_time, end_time)
        looking_total_time: total time in seconds
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    looking_frames = []
    segments = []

    current_segment_start = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            is_looking = False
            label = "No face"
            pitch = yaw = roll = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pose = _get_head_pose(frame, face_landmarks)

                if pose is not None:
                    pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs = pose

                    # Rough rule: facing camera if yaw/pitch are both small
                    if abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold:
                        is_looking = True
                        label = "Looking at camera"
                    else:
                        label = "Looking elsewhere"

                    # Optional: draw nose direction line
                    nose_end_3d = np.array([(0.0, 0.0, 100.0)])
                    nose_tip_3d = np.array([(0.0, 0.0, 0.0)])

                    nose_tip_2d, _ = cv2.projectPoints(
                        nose_tip_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    nose_end_2d, _ = cv2.projectPoints(
                        nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )

                    p1 = tuple(nose_tip_2d[0][0].astype(int))
                    p2 = tuple(nose_end_2d[0][0].astype(int))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)

            current_time = frame_idx / fps

            if is_looking:
                looking_frames.append(current_time)
                if current_segment_start is None:
                    current_segment_start = current_time
            else:
                if current_segment_start is not None:
                    segment_end = current_time
                    if segment_end - current_segment_start >= min_segment_duration:
                        segments.append((current_segment_start, segment_end))
                    current_segment_start = None

            # Draw label
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_looking else (0, 0, 255), 2)

            if pitch is not None and yaw is not None and roll is not None:
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            frame_idx += 1

    # Handle video ending while still in segment
    total_duration = frame_idx / fps
    if current_segment_start is not None:
        if total_duration - current_segment_start >= min_segment_duration:
            segments.append((current_segment_start, total_duration))

    cap.release()
    if writer is not None:
        writer.release()

    looking_total_time = sum(end - start for start, end in segments)

    # Print results
    print("\n Looking at camera time segments ")
    if not segments:
        print("No valid looking-at-camera segments found.")
    else:
        for i, (start, end) in enumerate(segments, 1):
            print(f"{i}. {start:.2f}s - {end:.2f}s  (duration: {end - start:.2f}s)")

    print(f"\nTotal looking-at-camera time: {looking_total_time:.2f} seconds")
    print(f"Video total duration: {total_duration:.2f} seconds")

    return segments, looking_total_time


def analyse_gaze_from_camera(
    camera_index=0,
    yaw_threshold=15,
    pitch_threshold=15
):
    """
    Analyse gaze/head direction from a webcam stream.
    Press q to quit the preview window.
    """
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {camera_index}")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            is_looking = False
            label = "No face"
            pitch = yaw = roll = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                pose = _get_head_pose(frame, face_landmarks)

                if pose is not None:
                    pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs = pose

                    if abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold:
                        is_looking = True
                        label = "Looking at camera"
                    else:
                        label = "Looking elsewhere"

                    nose_end_3d = np.array([(0.0, 0.0, 100.0)])
                    nose_tip_3d = np.array([(0.0, 0.0, 0.0)])

                    nose_tip_2d, _ = cv2.projectPoints(
                        nose_tip_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    nose_end_2d, _ = cv2.projectPoints(
                        nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )

                    p1 = tuple(nose_tip_2d[0][0].astype(int))
                    p2 = tuple(nose_end_2d[0][0].astype(int))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_looking else (0, 0, 255), 2)

            if pitch is not None and yaw is not None and roll is not None:
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera Gaze Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
