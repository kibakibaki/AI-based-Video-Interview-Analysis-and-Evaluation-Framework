# AI-based Video Interview Analysis and Evaluation Framework

This project is a prototype for video interview upload, gaze analysis, and eye-contact confidence scoring. The backend uses Flask, OpenCV, MediaPipe, GazeTracking, dlib, and NumPy.

## Environment Setup

The virtual environment is intentionally not committed to Git. After cloning the repository, create and activate your own local environment:

```bash
python3.11 -m venv backend/venv311
source backend/venv311/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

If `python3.11` is not available, install Python 3.11 first or use another compatible Python 3 version:

```bash
python3 -m venv backend/venv311
source backend/venv311/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

After activation, `python` and `pip` should point to the virtual environment.

This project installs `antoinelame/GazeTracking` directly from GitHub. It also depends on `dlib`; if `dlib` fails to install on macOS, install CMake first:

```bash
brew install cmake
pip install -r backend/requirements.txt
```

## Run The Upload Website

With the virtual environment activated:

```bash
python backend/app.py
```

Or run it directly without activation:

```bash
backend/venv311/bin/python backend/app.py
```

Then open:

```text
http://127.0.0.1:5001
```

Upload an MP4 or MOV video. The backend validates the video, runs gaze analysis, and returns the confidence report.

## Switch Camera Or Video Analysis

For local testing without the upload page, edit these variables in `backend/app.py`:

```python
APP_MODE = "analysis"
ANALYSIS_SOURCE = "camera"
```

Valid values:

- `APP_MODE = "server"` starts the upload website.
- `APP_MODE = "analysis"` runs local gaze analysis directly.
- `ANALYSIS_SOURCE = "camera"` analyses the webcam.
- `ANALYSIS_SOURCE = "video"` analyses `LOCAL_VIDEO_PATH`.

When `ANALYSIS_SOURCE = "video"`, update `LOCAL_VIDEO_PATH` in `backend/app.py` to point to the video you want to analyse.

## Project Structure

```text
backend/
  app.py                  Flask app and local analysis entry point
  requirements.txt        Python dependencies
  face_analysis/
    analysis_utils.py       Unified camera/video analysis flow and scoring loop
    gaze_utils.py           GazeTracking adapter for pupil/eye gaze detection
    head_pose_utils.py      Head pose estimation and head-facing-camera checks
    confidence_scoring.py   Eye-contact confidence scoring
    video_utils.py          Frame extraction and basic face detection helpers
frontend/
  index.html              Upload page
  script.js               Upload request and result display
  style.css               Page styling
data/
  video_sample/           Uploaded or sample videos
  output/                 Annotated analysis videos
```

## Notes

- Do not commit `backend/venv311/`; each user should create it locally.
- On macOS, camera analysis may require terminal or IDE camera permissions.
- The first MediaPipe import can be slow while local caches are created.
