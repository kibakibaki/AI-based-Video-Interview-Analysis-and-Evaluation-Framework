#!/usr/bin/env python3
"""
Desktop window labelling tool for sliding-window interview clips.

Run from the project root:

    python tools/labeling_app.py

Optional:

    python tools/labeling_app.py --filename sample1.mp4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ModuleNotFoundError as exc:
    print(
        "Tkinter is not available in this Python installation.\n"
        "On Homebrew Python 3.11, install it with:\n"
        "  brew install python-tk@3.11",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

import cv2
from PIL import Image, ImageTk

import create_label_sheets as sheets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "sample_vid"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
WINDOW_LABEL_PATH = LABEL_DIR / "manual_window_labels.csv"
VIDEO_LABEL_PATH = LABEL_DIR / "manual_video_labels.csv"

VIDEO_CANVAS_SIZE = (720, 405)
DEFAULT_WINDOW_SIZE = 3.0
DEFAULT_STEP_SIZE = 3.0
VALID_LABEL_VALUES = {"Y", "N"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return sheets.read_csv_rows(path)


def write_csv_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    sheets.write_csv_rows(path, rows, columns)


def sample_sort_key(path_or_name: Path | str) -> tuple[int, str]:
    name = path_or_name.name if isinstance(path_or_name, Path) else path_or_name
    stem = Path(name).stem
    if stem.startswith("sample") and stem[6:].isdigit():
        return (int(stem[6:]), name)
    return (10**9, name)


def available_filenames() -> list[str]:
    filenames = []
    seen_stems = set()
    video_paths = sorted(VIDEO_DIR.iterdir(), key=sample_sort_key) if VIDEO_DIR.exists() else []
    for path in video_paths:
        if path.suffix.lower() in {".mp4", ".mov"}:
            filenames.append(path.name)
            seen_stems.add(path.stem)

    for row in read_csv_rows(WINDOW_LABEL_PATH):
        filename = row.get("filename")
        if filename and Path(filename).stem not in seen_stems and filename not in filenames:
            filenames.append(filename)
            seen_stems.add(Path(filename).stem)

    return sorted(filenames, key=sample_sort_key)


def source_video_path(filename: str) -> Path:
    exact_path = VIDEO_DIR / filename
    if exact_path.exists():
        return exact_path

    stem = Path(filename).stem
    for extension in (".mp4", ".mov"):
        candidate = VIDEO_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    return exact_path


def label_filename_for(source_filename: str) -> str:
    return f"{Path(source_filename).stem}.mp4"


def has_valid_scores(row: dict[str, str]) -> bool:
    return all(row.get(field["name"], "") in VALID_LABEL_VALUES for field in sheets.LABEL_FIELDS)


def unlabelled_rows(filenames: list[str]) -> list[dict[str, str]]:
    filename_set = {label_filename_for(filename) for filename in filenames}
    return [
        row
        for row in read_csv_rows(WINDOW_LABEL_PATH)
        if row.get("filename") in filename_set and not has_valid_scores(row)
    ]


def get_video_metadata(filename: str) -> tuple[float, float, int]:
    video_path = source_video_path(filename)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()

    if duration <= 0:
        raise RuntimeError(f"Cannot read video duration: {video_path}")

    return duration, fps, int(frame_count)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def build_dataset_rows_for_video(
    filename: str,
    window_size: float,
    step_size: float,
) -> list[dict[str, str]]:
    duration, fps, _ = get_video_metadata(filename)
    label_filename = label_filename_for(filename)
    rows = []
    window_start = 0.0

    while window_start < duration:
        window_end = min(window_start + window_size, duration)
        if window_end <= window_start:
            break

        clip_id = sheets.clip_id_for(label_filename, f"{window_start:.2f}", f"{window_end:.2f}")
        window_duration = window_end - window_start
        row = {
            "clip_id": clip_id,
            "filename": label_filename,
            "source_csv": "",
            "clip_path": "",
            "window_start": str(round(window_start, 2)),
            "window_end": str(round(window_end, 2)),
            "window_duration": str(round(window_duration, 2)),
            "window_frames": str(round(window_duration * fps)),
        }
        for column in sheets.WINDOW_LABEL_COLUMNS:
            row.setdefault(column, "")
        rows.append(row)

        if window_end >= duration:
            break
        window_start += step_size

    return rows


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        label_filename_for(row.get("filename", "")),
        row.get("window_start", ""),
        row.get("window_end", ""),
    )


def ensure_label_rows(filename: str, window_size: float, step_size: float) -> list[dict[str, str]]:
    existing_rows = [sheets.normalise_existing_row(row) for row in read_csv_rows(WINDOW_LABEL_PATH)]
    existing_by_key = {row_key(row): row for row in existing_rows}
    generated_rows = build_dataset_rows_for_video(filename, window_size, step_size)
    label_filename = label_filename_for(filename)

    merged_rows = [
        row
        for row in existing_rows
        if label_filename_for(row.get("filename", "")) != label_filename
    ]
    fixed_columns = {
        "clip_id",
        "filename",
        "source_csv",
        "clip_path",
        "window_start",
        "window_end",
        "window_duration",
        "window_frames",
    }

    for generated in generated_rows:
        existing = existing_by_key.get(row_key(generated), {})
        merged = dict(generated)
        for column in sheets.WINDOW_LABEL_COLUMNS:
            if column not in fixed_columns and existing.get(column):
                merged[column] = existing[column]
        merged_rows.append(merged)

    write_csv_rows(WINDOW_LABEL_PATH, merged_rows, sheets.WINDOW_LABEL_COLUMNS)
    ensure_video_label_row(filename)
    return generated_rows


def ensure_video_label_row(filename: str) -> None:
    duration, _, _ = get_video_metadata(filename)
    label_filename = label_filename_for(filename)
    rows = [sheets.normalise_existing_row(row) for row in read_csv_rows(VIDEO_LABEL_PATH)]
    existing = None
    remaining_rows = []
    for row in rows:
        if label_filename_for(row.get("filename", "")) == label_filename:
            existing = row
        else:
            remaining_rows.append(row)

    row = {
        "filename": label_filename,
        "source_csv": "",
        "duration_seconds": str(round(duration, 2)),
        "duration_label": format_duration(duration),
        "overall_score": "",
        "confidence_label": "",
    }
    if existing:
        for column in sheets.VIDEO_LABEL_COLUMNS:
            if column not in row and existing.get(column):
                row[column] = existing[column]

    for column in sheets.VIDEO_LABEL_COLUMNS:
        row.setdefault(column, "")

    write_csv_rows(VIDEO_LABEL_PATH, remaining_rows + [row], sheets.VIDEO_LABEL_COLUMNS)


class WindowLabeler:
    def __init__(
        self,
        root: tk.Tk,
        filenames: list[str],
        window_size: float,
        step_size: float,
        sample_limit: int | None,
    ):
        self.root = root
        self.window_size = window_size
        self.step_size = step_size
        self.sample_limit = sample_limit
        self.filenames = filenames
        self.filename = filenames[0] if filenames else ""
        self.rows: list[dict[str, str]] = []
        self.all_rows: list[dict[str, str]] = []
        self.index = 0
        self.cap: cv2.VideoCapture | None = None
        self.playback_end_frame: int | None = None
        self.after_id: str | None = None
        self.frame_image = None

        self.label_vars = {field["name"]: tk.BooleanVar(value=False) for field in sheets.LABEL_FIELDS}
        self.quality_var = tk.StringVar(value="clear")
        self.notes_text: tk.Text | None = None

        self.build_ui()
        self.load_random_queue()
        self.load_current_window()

    def load_random_queue(self) -> None:
        for filename in self.filenames:
            ensure_label_rows(filename, self.window_size, self.step_size)
        self.all_rows = read_csv_rows(WINDOW_LABEL_PATH)
        self.rows = unlabelled_rows(self.filenames)
        random.shuffle(self.rows)
        if self.sample_limit is not None:
            self.rows = self.rows[:self.sample_limit]
        self.index = 0
        scope = self.filename if len(self.filenames) == 1 else f"{len(self.filenames)} videos"
        self.root.title(f"Window Labeling - Random from {scope}")

    def build_ui(self) -> None:
        self.root.title(f"Window Labeling - {self.filename}")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        title = ttk.Label(outer, text="Sliding Window Labeling", font=("Arial", 20, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.progress_label = ttk.Label(outer, text="")
        self.progress_label.grid(row=0, column=1, sticky="e", pady=(0, 12))

        video_frame = ttk.Frame(outer)
        video_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(1, weight=1)

        self.time_label = ttk.Label(video_frame, text="", font=("Arial", 13, "bold"))
        self.time_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.video_label = tk.Label(
            video_frame,
            bg="#111827",
            width=VIDEO_CANVAS_SIZE[0],
            height=VIDEO_CANVAS_SIZE[1],
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        controls = ttk.Frame(video_frame)
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure((0, 1), weight=1)
        ttk.Button(controls, text="Previous Window", command=self.previous_window).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(controls, text="Skip", command=self.skip).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(6, 0),
        )

        form = ttk.Frame(outer)
        form.grid(row=1, column=1, sticky="nsew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Quality").grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Combobox(
            form,
            textvariable=self.quality_var,
            values=("clear", "unclear", "invalid"),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=(0, 12))

        row_index = 1
        for field in sheets.LABEL_FIELDS:
            ttk.Label(form, text=field["label"]).grid(row=row_index, column=0, sticky="nw", pady=4)

            field_frame = ttk.Frame(form)
            field_frame.grid(row=row_index, column=1, sticky="ew", pady=4)
            field_frame.columnconfigure(0, weight=1)

            ttk.Label(
                field_frame,
                text=field["hint"],
                wraplength=360,
                foreground="#56657a",
            ).grid(row=0, column=0, sticky="w")
            ttk.Checkbutton(
                field_frame,
                variable=self.label_vars[field["name"]],
            ).grid(row=0, column=1, sticky="e", padx=(8, 0))
            row_index += 1

        ttk.Label(form, text="Notes").grid(row=row_index, column=0, sticky="nw", pady=(12, 4))
        self.notes_text = tk.Text(form, height=4, wrap="word")
        self.notes_text.grid(row=row_index, column=1, sticky="ew", pady=(12, 4))
        row_index += 1

        self.status_label = ttk.Label(form, text="", foreground="#3c475c")
        self.status_label.grid(row=row_index, column=0, columnspan=2, sticky="w", pady=(8, 8))
        row_index += 1

        ttk.Button(form, text="Submit & Next", command=self.submit).grid(
            row=row_index,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(4, 0),
        )

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", self.replay_from_key)
        self.root.bind("<Return>", self.submit_from_key)

    def load_current_window(self) -> None:
        self.stop_video()

        if self.index >= len(self.rows):
            remaining_rows = unlabelled_rows(self.filenames)
            if remaining_rows:
                self.progress_label.config(text="Session queue completed")
                self.time_label.config(text="Restart to reshuffle remaining windows")
                self.status_label.config(text="This random queue is finished. Reopen the tool for a fresh order.")
                self.video_label.config(image="", text="Session queue completed", fg="white")
            else:
                self.progress_label.config(text="All videos completed")
                self.time_label.config(text="All windows completed")
                self.status_label.config(text="All sample_vid windows are labelled.")
                self.video_label.config(image="", text="All windows completed", fg="white")
            return

        row = self.rows[self.index]
        self.filename = row["filename"]
        self.progress_label.config(
            text=f"{row['filename']} | Random window {self.index + 1} / {len(self.rows)}"
        )
        self.time_label.config(text=f"{row['window_start']}s - {row['window_end']}s")

        for field in sheets.LABEL_FIELDS:
            self.label_vars[field["name"]].set(row.get(field["name"], "") == "Y")
        self.quality_var.set(row.get("annotation_quality") or "clear")

        assert self.notes_text is not None
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert("1.0", row.get("notes", ""))

        self.status_label.config(text="Tick observed behaviours. Space = replay. Enter = submit.")
        self.replay()

    def replay_from_key(self, event=None) -> str:
        self.replay()
        return "break"

    def submit_from_key(self, event=None) -> str:
        self.submit()
        return "break"

    def replay(self) -> None:
        self.stop_video()
        if self.index >= len(self.rows):
            return

        row = self.rows[self.index]
        video_path = source_video_path(row["filename"])
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Cannot open video:\n{video_path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        start = float(row["window_start"])
        end = float(row["window_end"])
        start_frame = max(0, int(round(start * fps)))
        self.playback_end_frame = max(start_frame + 1, int(round(end * fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.show_next_frame()

    def show_next_frame(self) -> None:
        if self.cap is None:
            return

        if self.playback_end_frame is not None:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= self.playback_end_frame:
                self.stop_video()
                return

        ok, frame = self.cap.read()
        if not ok:
            self.stop_video()
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        delay_ms = max(15, int(1000 / fps))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(VIDEO_CANVAS_SIZE)

        canvas = Image.new("RGB", VIDEO_CANVAS_SIZE, "#111827")
        x = (VIDEO_CANVAS_SIZE[0] - image.width) // 2
        y = (VIDEO_CANVAS_SIZE[1] - image.height) // 2
        canvas.paste(image, (x, y))

        self.frame_image = ImageTk.PhotoImage(canvas)
        self.video_label.config(image=self.frame_image, text="")
        self.after_id = self.root.after(delay_ms, self.show_next_frame)

    def stop_video(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.playback_end_frame = None

    def skip(self) -> None:
        self.index = min(self.index + 1, len(self.rows))
        self.load_current_window()

    def previous_window(self) -> None:
        if self.index <= 0:
            self.status_label.config(text="Already at the first window for this video.")
            self.replay()
            return

        self.index -= 1
        self.load_current_window()

    def submit(self) -> None:
        if self.index >= len(self.rows):
            return

        labels = {}
        for field in sheets.LABEL_FIELDS:
            labels[field["name"]] = "Y" if self.label_vars[field["name"]].get() else "N"

        row = self.rows[self.index]
        target_key = row_key(row)
        notes = self.notes_text.get("1.0", tk.END).strip() if self.notes_text else ""

        for all_row in self.all_rows:
            if row_key(all_row) != target_key:
                continue

            for field_name, value in labels.items():
                all_row[field_name] = value
                row[field_name] = value
            all_row["annotation_quality"] = self.quality_var.get()
            all_row["notes"] = notes
            row["annotation_quality"] = self.quality_var.get()
            row["notes"] = notes
            break

        write_csv_rows(WINDOW_LABEL_PATH, self.all_rows, sheets.WINDOW_LABEL_COLUMNS)
        self.status_label.config(text=f"Saved window {self.index + 1}.")
        self.index += 1
        self.load_current_window()

    def close(self) -> None:
        self.stop_video()
        self.root.destroy()


def choose_filenames(args_filename: str | None) -> list[str]:
    filenames = available_filenames()
    if not filenames:
        raise RuntimeError("No videos found in sample_vid. Add a video first.")

    if args_filename:
        if args_filename in filenames:
            return [args_filename]

        matching_filename = next(
            (filename for filename in filenames if Path(filename).stem == Path(args_filename).stem),
            None,
        )
        if matching_filename is None:
            raise RuntimeError(f"{args_filename} was not found in sample_vid.")
        return [matching_filename]

    return filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the desktop window labelling tool.")
    parser.add_argument("--filename", help="Video filename to label, for example sample1.mp4.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--limit", type=int, help="Maximum number of random windows to label in this session.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        filenames = choose_filenames(args.filename)
        if args.window_size <= 0 or args.step_size <= 0:
            raise RuntimeError("--window-size and --step-size must be positive.")
        if args.limit is not None and args.limit <= 0:
            raise RuntimeError("--limit must be positive when provided.")
        for filename in filenames:
            ensure_label_rows(filename, args.window_size, args.step_size)

        rows = unlabelled_rows(filenames)
        if not rows:
            raise RuntimeError("No unlabelled window rows found.")

        scope = filenames[0] if len(filenames) == 1 else f"{len(filenames)} videos"
        session_count = min(len(rows), args.limit) if args.limit is not None else len(rows)
        print(f"Prepared {session_count} random windows from {len(rows)} unlabelled windows in {scope}.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    root = tk.Tk()
    WindowLabeler(root, filenames, args.window_size, args.step_size, args.limit)
    root.mainloop()


if __name__ == "__main__":
    main()
