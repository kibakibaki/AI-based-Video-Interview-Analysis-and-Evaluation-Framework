#!/usr/bin/env python3
"""
Generate automatic visual feature CSV files for videos in sample_vid/.

The script analyses each source video and writes:

    data/output/sample1.csv
    data/output/sample1_windows.csv

Window features use the Vision default 3 second window / 3 second step, matching
the manual labelling tool.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
VIDEO_DIR = PROJECT_ROOT / "sample_vid"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
ALLOWED_EXTENSIONS = {".mp4", ".mov"}

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
sys.path.insert(0, str(BACKEND_DIR))

import cv2  # noqa: E402

from Vision import analyse_gaze  # noqa: E402


def sample_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("sample") and stem[6:].isdigit():
        return (int(stem[6:]), path.name)
    return (10**9, path.name)


def label_filename_for(video_path: Path) -> str:
    return f"{video_path.stem}.mp4"


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Cannot read duration for video file: {video_path}")

    return frame_count / fps


def serialise_segments(segments):
    return [
        {
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
        }
        for start, end in segments
    ]


def write_summary_csv(filename: str, duration: float, analysis: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{Path(filename).stem}.csv"
    confidence_report = analysis["confidence_report"]
    features = confidence_report.get("features", {})

    row = {
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "duration_label": format_duration(duration),
        "overall_score": confidence_report.get("overall_score"),
        "confidence_label": confidence_report.get("label"),
        "looking_total_time": analysis["looking_total_time"],
        "looking_segments_count": len(analysis["segments"]),
    }

    for key, value in features.items():
        row[key] = repr(value) if isinstance(value, list) else value

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return csv_path


def write_window_csv(filename: str, analysis: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{Path(filename).stem}_windows.csv"
    window_rows = analysis["confidence_report"].get("window_features", [])

    if not window_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_file.write("filename\n")
        return csv_path

    rows = []
    for window_row in window_rows:
        row = {"filename": filename}
        row.update(window_row)
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def analyse_video(video_path: Path, frame_stride: int) -> tuple[Path, Path]:
    filename = label_filename_for(video_path)
    duration = get_video_duration(video_path)

    segments, looking_total_time, confidence_report = analyse_gaze(
        source_type="video",
        video_path=str(video_path),
        output_video_path=None,
        yaw_threshold=25,
        pitch_threshold=20,
        min_segment_duration=0.3,
        use_eye_gaze=True,
        analysis_frame_stride=frame_stride,
        show_preview=False,
        enable_confidence_scoring=True,
    )

    analysis = {
        "segments": serialise_segments(segments),
        "looking_total_time": round(looking_total_time, 2),
        "confidence_report": confidence_report,
        "duration_seconds": round(duration, 2),
        "duration_label": format_duration(duration),
    }

    return write_summary_csv(filename, duration, analysis), write_window_csv(filename, analysis)


def video_paths(filename: str | None) -> list[Path]:
    if filename:
        requested = VIDEO_DIR / filename
        if requested.exists():
            return [requested]

        stem = Path(filename).stem
        matches = sorted(
            path
            for path in VIDEO_DIR.iterdir()
            if path.stem == stem and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        if matches:
            return matches[:1]
        raise RuntimeError(f"{filename} was not found in {VIDEO_DIR}")

    return sorted(
        (
            path
            for path in VIDEO_DIR.iterdir()
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=sample_sort_key,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate automatic window features for sample videos.")
    parser.add_argument("--filename", help="Analyse one video, for example sample1.mp4.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Analyse one frame every N frames.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    paths = video_paths(args.filename)
    if not paths:
        raise SystemExit(f"No videos found in {VIDEO_DIR}")

    for index, video_path in enumerate(paths, 1):
        filename = label_filename_for(video_path)
        window_csv_path = OUTPUT_DIR / f"{Path(filename).stem}_windows.csv"
        if window_csv_path.exists() and not args.force:
            print(f"[{index}/{len(paths)}] Skipping {video_path.name}; output already exists.")
            continue

        print(f"[{index}/{len(paths)}] Analysing {video_path.name} -> {filename}", flush=True)
        try:
            summary_path, windows_path = analyse_video(video_path, args.frame_stride)
        except Exception as exc:
            print(f"Failed to analyse {video_path.name}: {exc}", file=sys.stderr)
            continue

        print(f"  Wrote {summary_path.relative_to(PROJECT_ROOT)}")
        print(f"  Wrote {windows_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
