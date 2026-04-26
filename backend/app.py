# app.py
from video_utils import extract_frames, detect_faces
from gaze_utils import analyse_gaze_from_camera, analyse_gaze_from_video

video_path = "../data/video_sample/sample1.mp4"
frames_folder = "../data/frames"
faces_folder = "../data/faces"
annotated_video = "../data/output/gaze_result.mp4"

# Step 1: Extract frames from video
# extract_frames(video_path, frames_folder)

# Step 2: Detect faces from video frames
# detect_faces(frames_folder, faces_folder)

# Step 3: Analyse gaze / head direction from video
# segments, total_time = analyse_gaze_from_video(
#     video_path=video_path,
#     output_video_path=annotated_video,
#     yaw_threshold=25,
#     pitch_threshold=20,
#     min_segment_duration=0.3
# )

# Temporary: analyse gaze / head direction from camera
analyse_gaze_from_camera(
    camera_index=0,
    yaw_threshold=25,
    pitch_threshold=20,
    use_eye_gaze=True,
    eye_yaw_weight=90
)

print("Done.")
