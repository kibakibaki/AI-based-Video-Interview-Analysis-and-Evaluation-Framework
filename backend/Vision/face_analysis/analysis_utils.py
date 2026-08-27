# analysis_utils.py
from collections import Counter

import cv2
import os
import mediapipe as mp
import time

from .confidence_scoring import ConfidenceScorer, FrameObservation, print_confidence_report
from .gaze_utils import PupilGazeTracker
from .head_pose_utils import (
    angle_distance_from_front,
    draw_head_direction,
    estimate_head_pose,
    is_head_facing_camera,
    signed_angle_from_front,
)
from .visual_features import VisualFeatureTracker

GAZE_RATIO_CENTER = 0.5
GAZE_DIRECTIONAL_RANGE = 0.15
ATTENTION_CENTER_LIMIT = 1.0
ATTENTION_REFERENCE_BIN_SIZE = 0.5
ATTENTION_REFERENCE_MIN_SAMPLES = 30


class AttentionReferenceTracker:
    def __init__(
        self,
        bin_size=ATTENTION_REFERENCE_BIN_SIZE,
        min_samples=ATTENTION_REFERENCE_MIN_SAMPLES,
        center_limit=ATTENTION_CENTER_LIMIT,
    ):
        self.bin_size = bin_size
        self.min_samples = min_samples
        self.center_limit = center_limit
        self.offset_bin_counts = Counter()
        self.samples = 0

    def update_and_check(
        self,
        pitch,
        yaw,
        yaw_threshold,
        pitch_threshold,
        gaze_observation,
    ):
        offsets = _attention_offsets(
            pitch,
            yaw,
            yaw_threshold,
            pitch_threshold,
            gaze_observation,
        )
        if offsets is None:
            return is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold)

        horizontal_offset, vertical_offset = offsets
        self.samples += 1
        self.offset_bin_counts[self._offset_bin(horizontal_offset, vertical_offset)] += 1

        reference_horizontal, reference_vertical = self.reference_offset()
        return (
            abs(horizontal_offset - reference_horizontal) <= self.center_limit
            and abs(vertical_offset - reference_vertical) <= self.center_limit
        )

    def reference_offset(self):
        if self.samples < self.min_samples or not self.offset_bin_counts:
            return 0.0, 0.0

        primary_bin, _ = self.offset_bin_counts.most_common(1)[0]
        return primary_bin[0] * self.bin_size, primary_bin[1] * self.bin_size

    def _offset_bin(self, horizontal_offset, vertical_offset):
        return (
            round(horizontal_offset / self.bin_size),
            round(vertical_offset / self.bin_size),
        )


def _face_mesh_solution():
    try:
        return mp.solutions.face_mesh
    except AttributeError as exc:
        raise RuntimeError(
            "Installed mediapipe package does not provide mp.solutions.face_mesh. "
            "Install the supported dependency versions with: "
            "`python -m pip install --force-reinstall -r backend/requirements.txt`"
        ) from exc


def _is_looking_at_camera(
    pitch,
    yaw,
    yaw_threshold,
    pitch_threshold,
    gaze_observation=None
):
    attention_centered = _is_attention_centered(
        pitch,
        yaw,
        yaw_threshold,
        pitch_threshold,
        gaze_observation,
    )
    if (
        gaze_observation is None
        or (
            gaze_observation.horizontal_ratio is None
            and gaze_observation.vertical_ratio is None
        )
    ):
        return is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold)

    return attention_centered


def _is_attention_centered(
    pitch,
    yaw,
    yaw_threshold,
    pitch_threshold,
    gaze_observation,
):
    if gaze_observation is None:
        return is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold)

    offsets = _attention_offsets(
        pitch,
        yaw,
        yaw_threshold,
        pitch_threshold,
        gaze_observation,
    )
    if offsets is None:
        return is_head_facing_camera(pitch, yaw, yaw_threshold, pitch_threshold)

    horizontal_offset, vertical_offset = offsets

    return (
        abs(horizontal_offset) <= ATTENTION_CENTER_LIMIT
        and abs(vertical_offset) <= ATTENTION_CENTER_LIMIT
    )


def _attention_offsets(pitch, yaw, yaw_threshold, pitch_threshold, gaze_observation):
    if pitch is None or yaw is None:
        return None

    gaze_horizontal_ratio = None
    gaze_vertical_ratio = None
    if gaze_observation is not None:
        gaze_horizontal_ratio = gaze_observation.horizontal_ratio
        gaze_vertical_ratio = gaze_observation.vertical_ratio

    return (
        _combined_horizontal_attention_offset(yaw, yaw_threshold, gaze_horizontal_ratio),
        _combined_vertical_attention_offset(pitch, pitch_threshold, gaze_vertical_ratio),
    )


def _combined_horizontal_attention_offset(yaw, yaw_threshold, gaze_horizontal_ratio):
    head_offset = _safe_normalized(yaw, yaw_threshold)
    gaze_offset = _gaze_ratio_offset(gaze_horizontal_ratio)
    if gaze_offset is None:
        return head_offset
    return head_offset + gaze_offset


def _combined_vertical_attention_offset(pitch, pitch_threshold, gaze_vertical_ratio):
    head_offset = _safe_normalized(signed_angle_from_front(pitch), pitch_threshold)
    gaze_offset = _gaze_ratio_offset(gaze_vertical_ratio)
    if gaze_offset is None:
        return head_offset
    return head_offset + gaze_offset


def _gaze_ratio_offset(ratio):
    if ratio is None:
        return None
    return (ratio - GAZE_RATIO_CENTER) / GAZE_DIRECTIONAL_RANGE


def _safe_normalized(value, threshold):
    if threshold <= 0:
        return value
    return value / threshold


def analyse_gaze(
    source_type,
    video_path=None,
    camera_index=0,
    output_video_path=None,
    yaw_threshold=15,
    pitch_threshold=15,
    min_segment_duration=0.3,
    use_eye_gaze=True,
    analysis_frame_stride=1,
    show_preview=None,
    enable_confidence_scoring=False
):
    """
    Analyse whether the person is roughly looking at their primary attention point from a video
    file or a webcam stream.

    Parameters:
        source_type: "video" or "camera"
        video_path: input video path when source_type is "video"
        camera_index: webcam index when source_type is "camera"
        output_video_path: optional annotated video output path
        yaw_threshold: horizontal tolerance used when estimating attention direction
        pitch_threshold: vertical tolerance used when estimating attention direction
        min_segment_duration: minimum duration (seconds) to keep
        use_eye_gaze: combine GazeTracking pupil direction with head pose
        analysis_frame_stride: analyse one frame every N frames
        show_preview: whether to show OpenCV preview window

    Returns:
        segments: list of (start_time, end_time)
        looking_total_time: total time in seconds
        analysis_report: observable visual features and optional legacy score
    """
    if source_type not in {"video", "camera"}:
        raise ValueError('source_type must be either "video" or "camera"')

    if show_preview is None:
        show_preview = source_type == "camera"
    if analysis_frame_stride < 1:
        raise ValueError("analysis_frame_stride must be >= 1")

    if source_type == "video":
        if video_path is None:
            raise ValueError("video_path is required when source_type is video")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        source_label = video_path
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_label = f"camera {camera_index}"

    mp_face_mesh = _face_mesh_solution()

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
    failed_camera_reads = 0
    black_camera_frames = 0
    max_failed_camera_reads = 100
    max_black_camera_frames = 100
    camera_error = None
    segments = []
    confidence_scorer = ConfidenceScorer() if enable_confidence_scoring else None
    pupil_gaze_tracker = PupilGazeTracker() if use_eye_gaze else None
    visual_feature_tracker = VisualFeatureTracker()
    attention_reference_tracker = AttentionReferenceTracker()

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
                if source_type == "camera":
                    failed_camera_reads += 1
                    if failed_camera_reads <= max_failed_camera_reads:
                        if show_preview:
                            if cv2.waitKey(50) & 0xFF == ord("q"):
                                break
                        else:
                            time.sleep(0.05)
                        continue
                break
            failed_camera_reads = 0

            if frame is None or frame.size == 0:
                frame_idx += 1
                continue

            if source_type == "camera":
                gray_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray_preview.mean() < 2:
                    black_camera_frames += 1
                    if show_preview:
                        preview = frame.copy()
                        cv2.putText(preview, f"Camera {camera_index}: black feed", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(preview, "Try another CAMERA_INDEX. Press q to quit.", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Gaze Analysis", preview)
                        if cv2.waitKey(50) & 0xFF == ord("q"):
                            break
                    else:
                        time.sleep(0.05)
                    if black_camera_frames >= max_black_camera_frames:
                        camera_error = (
                            f"{source_label} opened but returned a black feed. "
                            "This usually means OpenCV selected the wrong camera device. "
                            "Try running with another index, for example: "
                            "CAMERA_INDEX=1 ./run_app.sh"
                        )
                        break
                    continue
                black_camera_frames = 0

            if frame_idx % analysis_frame_stride != 0:
                if writer is not None:
                    writer.write(frame)
                frame_idx += 1
                continue

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

                    if attention_reference_tracker.update_and_check(
                        pitch,
                        yaw,
                        yaw_threshold,
                        pitch_threshold,
                        gaze_observation,
                    ):
                        is_looking = True
                        label = "Looking at primary point"
                    else:
                        label = "Looking away from primary point"

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
            visual_feature_tracker.update(
                current_time=current_time,
                face_detected=face_detected,
                looking_at_camera=is_looking,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                gaze_observation=gaze_observation,
            )

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
            if face_detected:
                ref_horizontal, ref_vertical = attention_reference_tracker.reference_offset()
                cv2.putText(frame, f"Primary ref: H {ref_horizontal:.1f}, V {ref_vertical:.1f}", (20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                text_y += 30

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

            analysed_frames = (frame_idx // analysis_frame_stride) + 1
            if analysed_frames % 100 == 0:
                print(
                    f"Analysed {analysed_frames} sampled frames "
                    f"({current_time:.1f}s)...",
                    flush=True,
                )

            frame_idx += 1

    if source_type == "camera" and (camera_error is not None or frame_idx == 0):
        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        if camera_error is not None:
            raise ValueError(camera_error)
        raise ValueError(
            f"Cannot read usable frames from {source_label}. "
            "The camera may be blocked, still starting, or returning a black feed."
        )

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
    visual_features = visual_feature_tracker.finish(total_duration)
    window_features = visual_feature_tracker.window_features(total_duration)

    print("\n Looking at primary attention point time segments ")
    if not segments:
        print("No valid primary-attention segments found.")
    else:
        for i, (start, end) in enumerate(segments, 1):
            print(f"{i}. {start:.2f}s - {end:.2f}s  (duration: {end - start:.2f}s)")

    print(f"\nTotal primary-attention time: {looking_total_time:.2f} seconds")
    print(f"Video total duration: {total_duration:.2f} seconds")

    analysis_report = {
        "features": visual_features,
        "window_features": window_features,
    }
    if confidence_scorer is not None:
        legacy_confidence_report = confidence_scorer.report()
        analysis_report["legacy_confidence_report"] = legacy_confidence_report
        print_confidence_report(legacy_confidence_report)

    return segments, looking_total_time, analysis_report
