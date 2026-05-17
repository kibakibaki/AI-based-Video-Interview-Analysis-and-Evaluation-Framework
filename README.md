# AI-based Video Interview Analysis and Evaluation Framework

This project is a prototype for video interview upload, gaze analysis, and eye-contact confidence scoring. The backend uses Flask, OpenCV, MediaPipe, GazeTracking, dlib, and NumPy.

## Environment Setup

The virtual environment is intentionally not committed to Git. After cloning the repository, create and activate your own local environment.

First, check whether Python 3.11 is available:

```bash
python3.11 --version
```

If this prints a version number, create the environment with:

```bash
python3.11 -m venv backend/venv311
source backend/venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -r backend/requirements-gaze.txt
```

If `python3.11` is not found on macOS, install it with Homebrew:

```bash
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv backend/venv311
source backend/venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -r backend/requirements-gaze.txt
```

If you are on an Intel Mac and Homebrew is installed under `/usr/local`, use this path instead:

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m venv backend/venv311
```

After activation, `python` should point to the virtual environment. Use `python -m pip` instead of plain `pip`; this avoids installing packages into the wrong Python environment.

If `backend/venv311` already exists but points to a missing Python executable, remove it and recreate it:

```bash
rm -rf backend/venv311
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv backend/venv311
source backend/venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -r backend/requirements-gaze.txt
```

This project installs `antoinelame/GazeTracking` directly from GitHub. Install it with `--no-deps` because the upstream package metadata currently asks for a newer NumPy than the MediaPipe version used by this project.

The project also depends on `dlib`; if `dlib` fails to install on macOS, install CMake first:

```bash
brew install cmake
python -m pip install -r backend/requirements.txt
python -m pip install --no-deps -r backend/requirements-gaze.txt
```

If analysis fails with `module 'mediapipe' has no attribute 'solutions'`, your environment has an incompatible MediaPipe release. Reinstall the pinned project dependencies:

```bash
source backend/venv311/bin/activate
python -m pip install --force-reinstall -r backend/requirements.txt
python -m pip install --force-reinstall --no-deps -r backend/requirements-gaze.txt
```

## Run The Upload Website

With the virtual environment activated:

```bash
python backend/app.py
```

Or run it directly from the project root:

```bash
./run_app.sh
```

Then open:

```text
http://127.0.0.1:5001
```

Upload an MP4 or MOV video. The backend validates the video, runs gaze analysis, and returns the confidence report.

## Run The Desktop Labelling Tool

The labelling tool is independent from the upload website. Use it to prepare manually labelled training data from videos in `sample_vid/`. It creates its own sliding windows and does not require the main app to analyse the video first.

From the project root:

```bash
./run_labeling_app.sh
```

Without `--filename`, the tool randomly samples unlabelled windows across all videos in `sample_vid/`. Windows already saved in `data/labels/manual_window_labels.csv` are skipped, and each new run reshuffles the remaining unlabelled windows.

To label only a random subset in one session:

```bash
./run_labeling_app.sh --limit 50
```

If the script reports that Tkinter is not available, install the desktop window support for Homebrew Python:

```bash
brew install python-tk@3.11
```

To label a specific video:

```bash
./run_labeling_app.sh --filename sample1.mp4
```

With `--filename`, the tool randomly samples only the unlabelled windows from that video.

By default, the tool uses 3 second windows with a 3 second step, so windows do not overlap during manual labelling. You can change this for dataset creation:

```bash
./run_labeling_app.sh --filename sample1.mp4 --window-size 6 --step-size 6
```

The tool opens a desktop window, plays each window directly from the source video in `sample_vid/`, and writes checkbox-based Y/N observation labels to:

```text
data/labels/manual_window_labels.csv
```

This labelling step is for dataset creation and model training data. It is not part of the main interview analysis app.

## Generate Automatic Training Features

To generate window-level visual features for videos in `sample_vid/`:

```bash
./run_analysis_features.sh
```

To analyse one video:

```bash
./run_analysis_features.sh --filename sample1.mp4
```

The script writes:

```text
data/output/sample1.csv
data/output/sample1_windows.csv
```

`*_windows.csv` contains the automatic 3 second window features used as model input `X`. These rows align with `data/labels/manual_window_labels.csv`.

For both training and future prediction, videos are not physically cut into 3 second video files. The analyser reads the original video, groups observations into 3 second windows in memory, and writes only tabular feature rows to CSV.

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
  Vision/
    face_analysis/
      analysis_utils.py       Unified camera/video analysis flow and scoring loop
      gaze_utils.py           GazeTracking adapter for pupil/eye gaze detection
      head_pose_utils.py      Head pose estimation and head-facing-camera checks
      visual_features.py      ML-ready visual feature aggregation
      confidence_scoring.py   Eye-contact confidence scoring
      video_utils.py          Frame extraction and basic face detection helpers
frontend/
  index.html              Upload page
  script.js               Upload request and result display
  style.css               Page styling
data/
  video_sample/           Uploaded videos from the main app
  output/                 Summary CSV and sliding-window CSV analysis results
  labels/                 Manual labels for model training
  window_clips/           Legacy/generated clips, not required by the current labelling tool
sample_vid/               Source videos for the standalone labelling tool
tools/
  create_label_sheets.py  Builds/refreshes CSV label sheets
  labeling_app.py         Desktop window tool for manual labels
```

## Notes

- Do not commit `backend/venv311/`; each user should create it locally.
- On macOS, camera analysis may require terminal or IDE camera permissions.
- The first MediaPipe import can be slow while local caches are created.

## Visual Features

The backend returns ML-ready visual features under:

```text
analysis.confidence_report.features
```

Current visual features include:

- eye contact ratio
- face visibility ratio
- gaze center ratio
- looking-away total time
- longest looking-away duration
- looking-away segments
- head pitch offset, yaw, and roll mean/std
- head movement mean delta
- head movement stability score
- blink count and blink rate per minute
- horizontal and vertical gaze ratio mean/std

For each uploaded video, the backend writes two CSV files:

```text
data/output/sample1.csv
data/output/sample1_windows.csv
```

`sample1.csv` contains one summary row for the whole video. `sample1_windows.csv`
contains sliding-window visual features without cutting the original video file.
The main analysis output and standalone manual labelling tool both use 3 second windows with a 3 second step so training features and labels align.
