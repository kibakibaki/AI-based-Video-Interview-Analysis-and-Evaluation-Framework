#!/usr/bin/env python3
"""
Build a window-level training dataset by joining automatic features (X)
with manual labels (Y).

Join key:
    filename + window_start + window_end
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import create_label_sheets as sheets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "data" / "labels" / "manual_window_labels.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "window_training_dataset.csv"

KEY_COLUMNS = ["filename", "window_start", "window_end"]
LEVEL_COLUMN = sheets.LOOK_AWAY_LEVEL_COLUMN
BINARY_LOOK_AWAY_COLUMN = "looking_away"
FREQUENT_LOOK_AWAY_COLUMN = "frequent_looking_away"
LABEL_COLUMNS = [field["name"] for field in sheets.LABEL_FIELDS]
OUTPUT_LABEL_COLUMNS = [
    LEVEL_COLUMN,
    BINARY_LOOK_AWAY_COLUMN,
    FREQUENT_LOOK_AWAY_COLUMN,
    *LABEL_COLUMNS,
]
LABEL_METADATA_COLUMNS = ["annotation_quality", "notes"]
VALID_LABEL_VALUES = {"Y", "N"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def normalise_time(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return value.strip() if value else ""


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        Path(row.get("filename", "")).with_suffix(".mp4").name,
        normalise_time(row.get("window_start", "")),
        normalise_time(row.get("window_end", "")),
    )


def is_complete_label_row(row: dict[str, str]) -> bool:
    return (
        row.get(LEVEL_COLUMN, "") in sheets.TRAINABLE_LOOK_AWAY_LEVEL_VALUES
        and all(row.get(column, "") in VALID_LABEL_VALUES for column in LABEL_COLUMNS)
    )


def has_any_label(row: dict[str, str]) -> bool:
    return (
        row.get(LEVEL_COLUMN, "") in sheets.LOOK_AWAY_LEVEL_VALUES
        or any(row.get(column, "") in VALID_LABEL_VALUES for column in LABEL_COLUMNS)
    )


def load_feature_rows(feature_dir: Path) -> list[dict[str, str]]:
    rows = []
    for csv_path in sorted(feature_dir.glob("*_windows.csv")):
        for row in read_csv_rows(csv_path):
            if all(row.get(column) for column in KEY_COLUMNS):
                rows.append(row)
    return rows


def load_label_rows(label_path: Path, allow_partial: bool) -> dict[tuple[str, str, str], dict[str, str]]:
    labels = {}
    for row in read_csv_rows(label_path):
        if not all(row.get(column) for column in KEY_COLUMNS):
            continue
        if allow_partial:
            if not has_any_label(row):
                continue
        elif not is_complete_label_row(row):
            continue
        labels[row_key(row)] = row
    return labels


def feature_columns(feature_rows: list[dict[str, str]]) -> list[str]:
    columns = []
    excluded = set(
        KEY_COLUMNS
        + OUTPUT_LABEL_COLUMNS
        + sheets.LEGACY_LOOK_AWAY_COLUMNS
        + LABEL_METADATA_COLUMNS
    )
    for row in feature_rows:
        for column in row:
            if column not in excluded and column not in columns:
                columns.append(column)
    return columns


def encode_label(value: str, keep_yn: bool) -> str:
    if keep_yn or value not in VALID_LABEL_VALUES:
        return value
    return "1" if value == "Y" else "0"


def build_rows(
    feature_rows: list[dict[str, str]],
    label_rows: dict[tuple[str, str, str], dict[str, str]],
    features: list[str],
    keep_yn: bool,
) -> list[dict[str, str]]:
    rows = []
    for feature_row in feature_rows:
        label_row = label_rows.get(row_key(feature_row))
        if label_row is None:
            continue

        row = {
            "filename": Path(feature_row.get("filename", "")).with_suffix(".mp4").name,
            "window_start": normalise_time(feature_row.get("window_start", "")),
            "window_end": normalise_time(feature_row.get("window_end", "")),
        }
        for column in features:
            row[column] = feature_row.get(column, "")
        level = label_row.get(LEVEL_COLUMN, "")
        row[LEVEL_COLUMN] = level
        row[BINARY_LOOK_AWAY_COLUMN] = (
            "1" if level in {"1", "2"} else "0" if level == "0" else ""
        )
        row[FREQUENT_LOOK_AWAY_COLUMN] = (
            "1" if level == "2" else "0" if level in {"0", "1"} else ""
        )
        for column in LABEL_COLUMNS:
            row[column] = encode_label(label_row.get(column, ""), keep_yn)
        for column in LABEL_METADATA_COLUMNS:
            row[column] = label_row.get(column, "")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join automatic window features X with manual labels Y.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--label-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--allow-partial", action="store_true", help="Include rows with at least one Y/N label.")
    parser.add_argument("--keep-yn", action="store_true", help="Keep labels as Y/N instead of converting Y=1, N=0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_rows = load_feature_rows(args.feature_dir)
    label_rows = load_label_rows(args.label_path, args.allow_partial)
    features = feature_columns(feature_rows)
    rows = build_rows(feature_rows, label_rows, features, args.keep_yn)
    columns = KEY_COLUMNS + features + OUTPUT_LABEL_COLUMNS + LABEL_METADATA_COLUMNS
    write_csv(args.output_path, rows, columns)

    print(f"Automatic feature rows found: {len(feature_rows)}")
    print(f"Manual label rows usable: {len(label_rows)}")
    print(f"Merged training rows written: {len(rows)}")
    print(f"Output: {args.output_path.relative_to(PROJECT_ROOT)}")
    if not feature_rows:
        print("No *_windows.csv files were found. Run ./run_labeling_app.sh or ./run_analysis_features.sh first.")
    if not label_rows:
        print(
            "No usable manual labels were found. The default build requires "
            "look_away_level 0, 1, or 2 and complete Y/N labels."
        )


if __name__ == "__main__":
    main()
