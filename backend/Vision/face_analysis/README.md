# Face Analysis Module

This folder contains the facial analysis pipeline used by the interview evaluation backend.

## Entry Point

`analyse_gaze` is exported from `Vision.face_analysis` and re-exported by `Vision.__init__` for `backend/app.py`.

```python
from Vision import analyse_gaze
```

## Included Analysis

- face visibility tracking
- head pose estimation with pitch, yaw, and roll
- camera-facing / looking-away segment detection
- pupil gaze direction through GazeTracking
- blink count and blink rate
- eye-contact confidence scoring
- ML-ready visual feature aggregation

## Files

- `analysis_utils.py`: unified camera/video analysis flow
- `gaze_utils.py`: GazeTracking adapter
- `head_pose_utils.py`: MediaPipe/OpenCV head pose estimation
- `visual_features.py`: visual feature aggregation for model input
- `confidence_scoring.py`: eye-contact scoring rubric
- `video_utils.py`: frame and face helper utilities
