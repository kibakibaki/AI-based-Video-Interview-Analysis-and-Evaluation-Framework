#!/usr/bin/env python3
"""Build fixed-length facial-feature sequences from existing labelled windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-vision-events-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/vision-events-v2-cache")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from vision_features import FEATURE_NAMES, extract_frame_features  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = PROJECT_ROOT / "sample_vid"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "manual_window_labels.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "look_away_sequences.npz"
VIDEO_SUFFIXES = {".mp4", ".mov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 10 Hz facial-feature sequences for manually labelled windows."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--filenames", nargs="*", help="Optional video names or stems to include.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def video_index(video_dir: Path) -> dict[str, Path]:
    videos = {}
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos[path.stem] = path
    return videos


def load_label_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            level = row.get("look_away_level", "")
            if row.get("annotation_quality") != "clear" or level not in {"0", "1", "2"}:
                continue
            start = float(row["window_start"])
            end = float(row["window_end"])
            if end <= start:
                continue
            rows.append(
                {
                    "filename": row["filename"],
                    "stem": Path(row["filename"]).stem,
                    "start": start,
                    "end": end,
                    "level": int(level),
                    "target": int(level != "0"),
                    "window_id": row.get("clip_id")
                    or f"{Path(row['filename']).stem}_{start:.2f}_{end:.2f}",
                }
            )
    return rows


def balanced_limit(rows: list[dict[str, object]], limit: int, seed: int) -> list[dict[str, object]]:
    if limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["target"])].append(row)
    for class_rows in by_class.values():
        rng.shuffle(class_rows)

    selected = []
    classes = sorted(by_class)
    while len(selected) < limit and any(by_class.values()):
        for class_value in classes:
            if by_class[class_value] and len(selected) < limit:
                selected.append(by_class[class_value].pop())
    return selected


def read_window_sequence(
    capture: cv2.VideoCapture,
    face_mesh,
    start: float,
    end: float,
    sequence_length: int,
) -> np.ndarray:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("Video FPS is unavailable.")

    sample_times = np.linspace(start, end, sequence_length, endpoint=False)
    target_frames = np.rint(sample_times * fps).astype(int)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_frames[0]))
    current_frame_index = int(target_frames[0])
    last_frame = None
    vectors = []

    for target_frame in target_frames:
        while current_frame_index <= target_frame:
            ok, frame = capture.read()
            if not ok:
                last_frame = None
                break
            last_frame = frame
            current_frame_index += 1
        vectors.append(extract_frame_features(last_frame, face_mesh))

    return np.stack(vectors).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0 or args.window_seconds <= 0:
        raise SystemExit("--sample-rate and --window-seconds must be positive.")

    videos = video_index(args.video_dir)
    rows = load_label_rows(args.labels)
    if args.filenames:
        requested_stems = {Path(filename).stem for filename in args.filenames}
        rows = [row for row in rows if row["stem"] in requested_stems]
    rows = [row for row in rows if row["stem"] in videos]
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise SystemExit("--max-windows must be positive.")
        rows = balanced_limit(rows, args.max_windows, args.seed)
    rows.sort(key=lambda row: (str(row["stem"]), float(row["start"])))
    if not rows:
        raise SystemExit("No usable labelled windows match the available videos.")

    sequence_length = int(round(args.sample_rate * args.window_seconds))
    sequences = []
    targets = []
    original_levels = []
    groups = []
    window_ids = []
    skipped = []

    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["stem"])].append(row)

    processed = 0
    face_mesh_solution = mp.solutions.face_mesh
    with face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for video_number, (stem, video_rows) in enumerate(sorted(grouped_rows.items()), 1):
            video_path = videos[stem]
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                skipped.extend(str(row["window_id"]) for row in video_rows)
                print(f"[{video_number}/{len(grouped_rows)}] Cannot open {video_path.name}; skipped.")
                continue

            print(
                f"[{video_number}/{len(grouped_rows)}] {video_path.name}: "
                f"{len(video_rows)} labelled windows",
                flush=True,
            )
            for row in video_rows:
                try:
                    sequence = read_window_sequence(
                        capture,
                        face_mesh,
                        float(row["start"]),
                        float(row["end"]),
                        sequence_length,
                    )
                except Exception as exc:
                    skipped.append(str(row["window_id"]))
                    print(f"  Skipped {row['window_id']}: {exc}")
                    continue
                sequences.append(sequence)
                targets.append(int(row["target"]))
                original_levels.append(int(row["level"]))
                groups.append(str(row["filename"]))
                window_ids.append(str(row["window_id"]))
                processed += 1
                if processed % 25 == 0:
                    print(f"  Extracted {processed}/{len(rows)} windows", flush=True)
            capture.release()

    if not sequences:
        raise SystemExit("Feature extraction produced no sequences.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sample_rate_hz": args.sample_rate,
        "window_seconds": args.window_seconds,
        "sequence_length": sequence_length,
        "target": "binary",
        "target_names": ["no_gaze_shift", "gaze_shift"],
        "source_labels": str(args.labels),
        "skipped_windows": skipped,
    }
    np.savez_compressed(
        args.output,
        X=np.stack(sequences),
        y=np.asarray(targets, dtype=np.int64),
        original_levels=np.asarray(original_levels, dtype=np.int64),
        groups=np.asarray(groups),
        window_ids=np.asarray(window_ids),
        feature_names=np.asarray(FEATURE_NAMES),
        metadata=np.asarray(json.dumps(metadata)),
    )

    class_counts = np.bincount(np.asarray(targets), minlength=len(metadata["target_names"]))
    face_visibility = float(np.nanmean(np.stack(sequences)[:, :, 0]))
    print(f"Saved: {args.output}")
    print(f"Sequences: {len(sequences)} | shape: {np.stack(sequences).shape}")
    print(f"Videos: {len(set(groups))} | class counts: {class_counts.tolist()}")
    print(f"Mean sampled-frame face visibility: {face_visibility:.3f}")
    print(f"Skipped windows: {len(skipped)}")


if __name__ == "__main__":
    main()
