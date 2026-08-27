import csv
import os
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from Vision import analyse_gaze
from Vision.face_analysis.gaze_shift_classifier import (
    GazeShiftModelContractError,
    GazeShiftModelUnavailable,
    classify_gaze_shift_video,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
APP_DATA_DIR = PROJECT_DIR / "data" / "app_output"
UPLOAD_DIR = APP_DATA_DIR / "uploads"
OUTPUT_DIR = APP_DATA_DIR / "analysis"

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_DURATION_SECONDS = 5 * 60

# Change this variable to choose what happens when running this file directly.
# "server" starts the upload website; "analysis" runs the local gaze analysis.
APP_MODE = os.environ.get("APP_MODE", "server")
# Change this variable to switch local analysis input.
# Valid values: "camera", "single_camera", or "video".
ANALYSIS_SOURCE = "camera"

LOCAL_VIDEO_PATH = PROJECT_DIR / "data" / "video_sample" / "sample1.mp4"

# OpenCV camera indexes on macOS depend on the connected devices. Override with:
# CAMERA_INDEX=2 ./run_app.sh
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
CAMERA_INDEX_CANDIDATES = [CAMERA_INDEX]

app = Flask(__name__, static_folder=None)


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return None

    return frame_count / fps


def _format_duration(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _serialise_segments(segments):
    return [
        {
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
        }
        for start, end in segments
    ]


def _new_upload_filename(extension):
    """Return a UUID filename that cannot collide with an existing upload."""
    while True:
        filename = f"{uuid4()}{extension}"
        if not (UPLOAD_DIR / filename).exists():
            return filename


def _write_analysis_csv(filename, duration, analysis):
    csv_path = OUTPUT_DIR / f"{Path(filename).stem}.csv"
    gaze_shift_report = analysis["gaze_shift_report"]
    summary = gaze_shift_report["summary"]
    features = analysis.get("visual_features", {})

    row = {
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "duration_label": _format_duration(duration),
        "task": gaze_shift_report["task"],
        "model": gaze_shift_report["model"],
        **summary,
        "looking_total_time": analysis["looking_total_time"],
        "looking_segments_count": len(analysis["segments"]),
    }

    for key, value in features.items():
        if isinstance(value, list):
            row[key] = repr(value)
        else:
            row[key] = value

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return csv_path


def _write_window_features_csv(filename, analysis):
    feature_rows = analysis.get("window_features", [])
    prediction_rows = analysis["gaze_shift_report"].get("windows", [])
    csv_path = OUTPUT_DIR / f"{Path(filename).stem}_windows.csv"

    if not prediction_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_file.write("filename\n")
        return csv_path

    features_by_window = {
        (row.get("window_start"), row.get("window_end")): row
        for row in feature_rows
    }
    rows = []
    for prediction in prediction_rows:
        key = (prediction["window_start"], prediction["window_end"])
        probabilities = prediction.get("probabilities", {})
        row = {
            "filename": filename,
            **features_by_window.get(key, {}),
            "window_start": prediction["window_start"],
            "window_end": prediction["window_end"],
            "gaze_state": prediction["state"],
            "decision_reason": prediction.get("reason"),
            "model_face_visibility_ratio": prediction.get("face_visibility_ratio"),
            "model_visual_observability_ratio": prediction.get(
                "visual_observability_ratio"
            ),
            "prediction_probability": prediction.get("prediction_probability"),
            "probability_no_gaze_shift": probabilities.get("no_gaze_shift"),
            "probability_gaze_shift": probabilities.get("gaze_shift"),
        }
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _build_analysis_response(filename, saved_path, duration):
    gaze_shift_report = classify_gaze_shift_video(saved_path)
    segments, looking_total_time, visual_report = analyse_gaze(
        source_type="video",
        video_path=str(saved_path),
        output_video_path=None,
        yaw_threshold=25,
        pitch_threshold=20,
        min_segment_duration=0.3,
        use_eye_gaze=True,
        analysis_frame_stride=10,
        show_preview=False,
        enable_confidence_scoring=False,
    )
    analysis = {
        "segments": _serialise_segments(segments),
        "looking_total_time": round(looking_total_time, 2),
        "visual_features": visual_report.get("features", {}),
        "window_features": visual_report.get("window_features", []),
        "gaze_shift_report": gaze_shift_report,
        "duration_seconds": round(duration, 2),
        "duration_label": _format_duration(duration),
    }
    csv_path = _write_analysis_csv(filename, duration, analysis)
    window_csv_path = _write_window_features_csv(filename, analysis)
    analysis["csv_path"] = str(csv_path.relative_to(PROJECT_DIR))
    analysis["window_csv_path"] = str(window_csv_path.relative_to(PROJECT_DIR))

    return analysis


def run_local_analysis():
    if ANALYSIS_SOURCE == "camera":
        last_error = None
        for camera_index in CAMERA_INDEX_CANDIDATES:
            try:
                print(f"Trying camera {camera_index}...", flush=True)
                return analyse_gaze(
                    source_type="camera",
                    camera_index=camera_index,
                    yaw_threshold=25,
                    pitch_threshold=20,
                    use_eye_gaze=True,
                    show_preview=True,
                    enable_confidence_scoring=False,
                )
            except ValueError as exc:
                last_error = exc
                print(f"Camera {camera_index} unavailable: {exc}", flush=True)

        raise RuntimeError(
            "No camera could be opened. Check macOS Camera permissions for the "
            "app running this command, close other camera apps, or update "
            f"CAMERA_INDEX_CANDIDATES. Last error: {last_error}"
        )

    if ANALYSIS_SOURCE == "single_camera":
        return analyse_gaze(
            source_type="camera",
            camera_index=CAMERA_INDEX,
            yaw_threshold=25,
            pitch_threshold=20,
            use_eye_gaze=True,
            show_preview=True,
            enable_confidence_scoring=False,
        )

    if ANALYSIS_SOURCE == "video":
        return analyse_gaze(
            source_type="video",
            video_path=str(LOCAL_VIDEO_PATH),
            output_video_path=None,
            yaw_threshold=25,
            pitch_threshold=20,
            min_segment_duration=0.3,
            use_eye_gaze=True,
            analysis_frame_stride=5,
            show_preview=False,
            enable_confidence_scoring=False,
        )

    raise ValueError('ANALYSIS_SOURCE must be "camera", "single_camera", or "video"')


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/script.js")
def script():
    return send_from_directory(FRONTEND_DIR, "script.js")


@app.get("/style.css")
def style():
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.post("/upload")
def upload_video():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No video file was provided."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"ok": False, "error": "Please choose a video file first."}), 400

    original_name = video_file.filename
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "ok": False,
            "error": "Unsupported file format. Please upload an MP4 or MOV video."
        }), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = _new_upload_filename(extension)
    saved_path = UPLOAD_DIR / filename
    video_file.save(saved_path)

    duration = _get_video_duration(saved_path)
    if duration is None:
        saved_path.unlink(missing_ok=True)
        return jsonify({
            "ok": False,
            "error": "The uploaded file could not be read as a valid video."
        }), 400

    if duration > MAX_DURATION_SECONDS:
        saved_path.unlink(missing_ok=True)
        return jsonify({
            "ok": False,
            "error": "Video is too long. Maximum duration is 5 minutes.",
            "duration_seconds": round(duration, 2),
            "duration_label": _format_duration(duration)
        }), 400

    return jsonify({
        "ok": True,
        "message": "Video uploaded and validated successfully.",
        "filename": filename,
        "path": str(saved_path.relative_to(PROJECT_DIR)),
        "duration_seconds": round(duration, 2),
        "duration_label": _format_duration(duration),
        "next_step": "Ready for analysis."
    })


@app.post("/analyse")
def analyse_uploaded_video():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "No uploaded filename was provided."}), 400

    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return jsonify({"ok": False, "error": "Invalid uploaded filename."}), 400

    saved_path = UPLOAD_DIR / filename
    if not saved_path.exists():
        return jsonify({"ok": False, "error": "Uploaded video was not found."}), 404

    duration = _get_video_duration(saved_path)
    if duration is None:
        return jsonify({
            "ok": False,
            "error": "The uploaded file could not be read as a valid video."
        }), 400

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        analysis = _build_analysis_response(filename, saved_path, duration)
    except (GazeShiftModelUnavailable, GazeShiftModelContractError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503

    return jsonify({
        "ok": True,
        "message": "Video analysis completed successfully.",
        "filename": filename,
        "analysis": analysis,
        "next_step": "Analysis complete."
    })


def run_server():
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=True, host="127.0.0.1", port=port)


if __name__ == "__main__":
    if APP_MODE == "server":
        run_server()
    elif APP_MODE == "analysis":
        run_local_analysis()
    else:
        raise ValueError('APP_MODE must be either "server" or "analysis"')
