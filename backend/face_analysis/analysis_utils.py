# analysis_utils.py
import cv2
import os
import mediapipe as mp

from .confidence_scoring import ConfidenceScorer, FrameObservation, print_confidence_report
from .gaze_utils import PupilGazeTracker
from .head_pose_utils import (
    angle_distance_from_front,
    draw_head_direction,
    estimate_head_pose,
    is_head_facing_camera,
)


def _is_looking_at_camera(
    pitch,
    yaw,
    yaw_threshold,
    pitch_threshold,
    gaze_observation=None
):
    head_facing = is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold)
    if gaze_observation is None or gaze_observation.direction is None:
        return head_facing

    return head_facing and gaze_observation.is_center


def analyse_gaze(
    source_type,
    video_path=None,
    camera_index=0,
    output_video_path=None,
    yaw_threshold=15,
    pitch_threshold=15,
    min_segment_duration=0.3,
    use_eye_gaze=True,
    show_preview=None,
    enable_confidence_scoring=True
):
    """
    Analyse whether the person is roughly looking at the camera from a video
    file or a webcam stream.

    Parameters:
        source_type: "video" or "camera"
        video_path: input video path when source_type is "video"
        camera_index: webcam index when source_type is "camera"
        output_video_path: optional annotated video output path
        yaw_threshold: abs(yaw) <= threshold => looking at camera
        pitch_threshold: abs(pitch) <= threshold => looking at camera
        min_segment_duration: minimum duration (seconds) to keep
        use_eye_gaze: combine GazeTracking pupil direction with head pose
        show_preview: whether to show OpenCV preview window

    Returns:
        segments: list of (start_time, end_time)
        looking_total_time: total time in seconds
        confidence_report: scoring dictionary or None
    """
    if source_type not in {"video", "camera"}:
        raise ValueError('source_type must be either "video" or "camera"')

    if show_preview is None:
        show_preview = source_type == "camera"

    if source_type == "video":
        if video_path is None:
            raise ValueError("video_path is required when source_type is video")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        source_label = video_path
    else:
        cap = cv2.VideoCapture(camera_index)
        source_label = f"camera {camera_index}"

    mp_face_mesh = mp.solutions.face_mesh

    if not cap.isOpened():
        raise ValueError(f"Cannot open {source_label}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path is not None:
        output_dir = os.path.dirname(output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    segments = []
    confidence_scorer = ConfidenceScorer() if enable_confidence_scoring else None
    pupil_gaze_tracker = PupilGazeTracker() if use_eye_gaze else None

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
            gaze_observation = None
            face_detected = False

            if pupil_gaze_tracker is not None:
                gaze_observation = pupil_gaze_tracker.analyse_frame(frame)

            if results.multi_face_landmarks:
                face_detected = True
                face_landmarks = results.multi_face_landmarks[0]
                pose = estimate_head_pose(frame, face_landmarks)

                if pose is not None:
                    pitch, yaw, roll, rvec, tvec, camera_matrix, dist_coeffs = pose

                    if _is_looking_at_camera(
                        pitch,
                        yaw,
                        yaw_threshold,
                        pitch_threshold,
                        gaze_observation=gaze_observation
                    ):
                        is_looking = True
                        label = "Looking at camera"
                    else:
                        label = "Looking elsewhere"

                    draw_head_direction(frame, rvec, tvec, camera_matrix, dist_coeffs)

            current_time = frame_idx / fps

            if is_looking:
                if current_segment_start is None:
                    current_segment_start = current_time
            else:
                if current_segment_start is not None:
                    segment_end = current_time
                    if segment_end - current_segment_start >= min_segment_duration:
                        segments.append((current_segment_start, segment_end))
                    current_segment_start = None

            if confidence_scorer is not None:
                confidence_scorer.update(FrameObservation(
                    face_detected=face_detected,
                    looking_at_camera=is_looking
                ))

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
                cv2.putText(frame, f"Pitch offset: {angle_distance_from_front(pitch):.1f}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            text_y = 200
            if gaze_observation is not None:
                gaze_label = gaze_observation.direction or "unknown"
                cv2.putText(frame, f"Gaze: {gaze_label}", (20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if gaze_observation.horizontal_ratio is not None:
                    cv2.putText(frame, f"Gaze H: {gaze_observation.horizontal_ratio:.2f}", (20, text_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    text_y += 30
                if gaze_observation.left_pupil is not None:
                    cv2.circle(frame, gaze_observation.left_pupil, 3, (0, 255, 255), -1)
                if gaze_observation.right_pupil is not None:
                    cv2.circle(frame, gaze_observation.right_pupil, 3, (0, 255, 255), -1)
                text_y += 30

            if confidence_scorer is not None:
                cv2.putText(frame, f"Confidence: {confidence_scorer.overall_score:.1f}/100", (20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, confidence_scorer.label, (20, text_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            if show_preview:
                cv2.imshow("Gaze Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    total_duration = frame_idx / fps
    if current_segment_start is not None:
        if total_duration - current_segment_start >= min_segment_duration:
            segments.append((current_segment_start, total_duration))

    cap.release()
    if writer is not None:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    looking_total_time = sum(end - start for start, end in segments)

    print("\n Looking at camera time segments ")
    if not segments:
        print("No valid looking-at-camera segments found.")
    else:
        for i, (start, end) in enumerate(segments, 1):
            print(f"{i}. {start:.2f}s - {end:.2f}s  (duration: {end - start:.2f}s)")

    print(f"\nTotal looking-at-camera time: {looking_total_time:.2f} seconds")
    print(f"Video total duration: {total_duration:.2f} seconds")

    confidence_report = None
    if confidence_scorer is not None:
        confidence_report = confidence_scorer.report()
        print_confidence_report(confidence_report)

    return segments, looking_total_time, confidence_report
