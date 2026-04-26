import os
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
UPLOAD_DIR = PROJECT_DIR / "data" / "video_sample"

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_DURATION_SECONDS = 5 * 60


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
    filename = secure_filename(original_name)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=True, host="127.0.0.1", port=port)
