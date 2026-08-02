#!/usr/bin/env python3
"""
Create or refresh manual annotation sheet columns.

The script is intentionally dependency-free so it can run in a fresh clone:

    python tools/create_label_sheets.py

Existing manual annotations are preserved when rows can be matched by video
filename or by the window key: filename + window_start + window_end.

The desktop labelling app can create window rows directly from videos. This
script is still useful when you want to align labels with generated analysis
CSV files under data/output/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_LABEL_DIR = PROJECT_ROOT / "data" / "labels"
DEFAULT_CLIP_ROOT = PROJECT_ROOT / "data" / "window_clips"

LOOK_AWAY_LEVEL_COLUMN = "look_away_level"
LOOK_AWAY_LEVEL_VALUES = {"0", "1", "2", "U"}
TRAINABLE_LOOK_AWAY_LEVEL_VALUES = {"0", "1", "2"}
LEGACY_SEVERITY_COLUMN = "look_away_severity"
LEGACY_LOOK_AWAY_COLUMNS = ["looking_away", "frequent_looking_away"]

LABEL_FIELDS = [
    {
        "name": "face_not_visible",
        "label": "Face Not Visible",
        "hint": "Y if the face is missing, blocked, or too unclear to judge.",
    },
    {
        "name": "excessive_head_movement",
        "label": "Excessive Head Movement",
        "hint": "Y if head motion is noticeably unstable or distracting.",
    },
    {
        "name": "noticeable_fidgeting",
        "label": "Noticeable Fidgeting",
        "hint": "Y if there is visible restless movement or fidgeting.",
    },
    {
        "name": "distracted_by_offscreen_activity",
        "label": "Offscreen Distraction",
        "hint": "Y if the candidate appears distracted by something offscreen.",
    },
    {
        "name": "camera_or_video_issue",
        "label": "Camera Or Video Issue",
        "hint": "Y if blur, freeze, lighting, framing, or camera movement affects the clip.",
    },
    {
        "name": "reading_from_screen_or_notes",
        "label": "Reading Notes",
        "hint": "Y if the candidate appears to read from notes, a script, or another screen.",
    },
    {
        "name": "looking_down_frequently",
        "label": "Looking Down",
        "hint": "Y if the candidate repeatedly looks downward.",
    },
    {
        "name": "body_out_of_frame",
        "label": "Body Out Of Frame",
        "hint": "Y if the head or upper body moves partly out of the visible frame.",
    },
    {
        "name": "poor_lighting",
        "label": "Poor Lighting",
        "hint": "Y if the lighting is too dark, too bright, or uneven enough to affect judgement.",
    },
    {
        "name": "background_distraction_visible",
        "label": "Background Distraction",
        "hint": "Y if the background contains a noticeable visual distraction.",
    },
    {
        "name": "hand_to_face_or_hair",
        "label": "Hand To Face Or Hair",
        "hint": "Y if the candidate touches their face, hair, glasses, or headset.",
    },
]


VIDEO_LABEL_COLUMNS = [
    "filename",
    "duration_seconds",
    "duration_label",
    "overall_score",
    *LEGACY_LOOK_AWAY_COLUMNS,
    *[field["name"] for field in LABEL_FIELDS],
    "annotation_quality",
    "notes",
]


WINDOW_LABEL_COLUMNS = [
    "clip_id",
    "filename",
    "clip_path",
    "window_start",
    "window_end",
    "window_duration",
    "is_labeled",
    LOOK_AWAY_LEVEL_COLUMN,
    *LEGACY_LOOK_AWAY_COLUMNS,
    *[field["name"] for field in LABEL_FIELDS],
    "annotation_quality",
    "notes",
]


LEGACY_COLUMN_ALIASES = {
    "manual_score": "confidence_score",
    "manual_confidence_score": "confidence_score",
    "manual_focus_score": "focus_score",
    "manual_engagement_score": "engagement_score",
    "head_stability": "head_stability_score",
    "eye_contact_quality": "eye_contact_score",
}

REVERSED_LEGACY_COLUMN_ALIASES = {
    "manual_nervousness_score": "calmness_score",
    "nervousness_score": "calmness_score",
    "facial_tension_score": "facial_relaxation_score",
    "reading_script_score": "spontaneity_score",
    "restlessness_score": "composure_score",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalise_existing_row(row: dict[str, str]) -> dict[str, str]:
    normalised = dict(row)
    if not normalised.get(LOOK_AWAY_LEVEL_COLUMN):
        legacy_looking_away = normalised.get("looking_away", "")
        legacy_frequent = normalised.get("frequent_looking_away", "")
        legacy_severity = normalised.get(LEGACY_SEVERITY_COLUMN, "")

        if legacy_frequent == "Y":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "2"
        elif legacy_looking_away == "Y":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "1"
        elif legacy_looking_away == "N" and legacy_frequent == "N":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "0"
        elif legacy_severity == "U":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "U"
        elif legacy_severity == "2":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "1"
        elif legacy_severity == "0":
            normalised[LOOK_AWAY_LEVEL_COLUMN] = "0"
    for old_column, new_column in LEGACY_COLUMN_ALIASES.items():
        if not normalised.get(new_column) and normalised.get(old_column):
            normalised[new_column] = normalised[old_column]
    for old_column, new_column in REVERSED_LEGACY_COLUMN_ALIASES.items():
        if not normalised.get(new_column) and normalised.get(old_column):
            normalised[new_column] = reverse_score(normalised[old_column])
    return normalised


def reverse_score(value: str) -> str:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return ""
    if score < 1 or score > 5:
        return ""
    return str(6 - score)


def existing_video_labels(path: Path) -> dict[str, dict[str, str]]:
    return {
        row.get("filename", ""): normalise_existing_row(row)
        for row in read_csv_rows(path)
        if row.get("filename")
    }


def existing_window_labels(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    labels = {}
    for row in read_csv_rows(path):
        if not row.get("filename"):
            continue
        normalised = normalise_existing_row(row)
        key = (
            normalised.get("filename", ""),
            normalised.get("window_start", ""),
            normalised.get("window_end", ""),
        )
        labels[key] = normalised
    return labels


def summary_csv_paths(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.csv")
        if not path.name.endswith("_windows.csv")
    )


def window_csv_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*_windows.csv"))


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def clip_id_for(filename: str, window_start: str, window_end: str) -> str:
    stem = Path(filename).stem
    start = float(window_start)
    end = float(window_end)
    return f"{stem}_{start:07.2f}_{end:07.2f}"


def clip_path_for(filename: str, window_start: str, window_end: str, clip_root: Path) -> str:
    clip_id = clip_id_for(filename, window_start, window_end)
    clip_path = clip_root / Path(filename).stem / f"{clip_id}.mp4"
    return relative_path(clip_path)


def build_video_rows(output_dir: Path, label_path: Path) -> list[dict[str, str]]:
    existing_labels = existing_video_labels(label_path)
    rows = []

    for source_csv in summary_csv_paths(output_dir):
        source_rows = read_csv_rows(source_csv)
        if not source_rows:
            continue

        source_row = source_rows[0]
        filename = source_row.get("filename", "")
        existing = existing_labels.get(filename, {})

        row = {
            "filename": filename,
            "source_csv": relative_path(source_csv),
            "duration_seconds": source_row.get("duration_seconds", ""),
            "duration_label": source_row.get("duration_label", ""),
            "overall_score": source_row.get("overall_score", ""),
            "confidence_label": source_row.get("confidence_label", ""),
        }
        row.update(preserved_label_values(existing, VIDEO_LABEL_COLUMNS))
        rows.append(row)

    return rows


def build_window_rows(output_dir: Path, label_path: Path, clip_root: Path) -> list[dict[str, str]]:
    existing_labels = existing_window_labels(label_path)
    rows = []

    for source_csv in window_csv_paths(output_dir):
        for source_row in read_csv_rows(source_csv):
            filename = source_row.get("filename", "")
            window_start = source_row.get("window_start", "")
            window_end = source_row.get("window_end", "")
            if not filename or not window_start or not window_end:
                continue

            existing = existing_labels.get((filename, window_start, window_end), {})
            row = {
                "clip_id": clip_id_for(filename, window_start, window_end),
                "filename": filename,
                "source_csv": relative_path(source_csv),
                "clip_path": clip_path_for(filename, window_start, window_end, clip_root),
                "window_start": window_start,
                "window_end": window_end,
            }

            for column in WINDOW_LABEL_COLUMNS:
                if column not in row:
                    row[column] = source_row.get(column, "")

            row.update(preserved_label_values(existing, WINDOW_LABEL_COLUMNS))
            rows.append(row)

    return rows


def preserved_label_values(existing: dict[str, str], columns: list[str]) -> dict[str, str]:
    fixed_columns = {
        "clip_id",
        "filename",
        "source_csv",
        "clip_path",
        "duration_seconds",
        "duration_label",
        "overall_score",
        "confidence_label",
        "window_start",
        "window_end",
        "window_duration",
        "window_frames",
        "eye_contact_ratio",
        "face_visibility_ratio",
        "gaze_center_ratio",
        "looking_away_total_time",
        "head_pitch_offset_mean",
        "head_yaw_mean",
        "head_yaw_std",
        "head_roll_mean",
        "head_roll_std",
        "head_movement_mean_delta",
        "head_movement_stability_score",
        "blink_count",
        "blink_rate_per_minute",
    }
    return {
        column: existing.get(column, "")
        for column in columns
        if column not in fixed_columns
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manual video/window label sheets from analysis CSV output."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_label_path = args.label_dir / "manual_video_labels.csv"
    window_label_path = args.label_dir / "manual_window_labels.csv"

    video_rows = build_video_rows(args.output_dir, video_label_path)
    window_rows = build_window_rows(args.output_dir, window_label_path, args.clip_root)

    write_csv_rows(video_label_path, video_rows, VIDEO_LABEL_COLUMNS)
    write_csv_rows(window_label_path, window_rows, WINDOW_LABEL_COLUMNS)

    print(f"Wrote {len(video_rows)} video label rows to {relative_path(video_label_path)}")
    print(f"Wrote {len(window_rows)} window label rows to {relative_path(window_label_path)}")


if __name__ == "__main__":
    main()
