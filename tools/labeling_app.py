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

import analyse_sample_videos as automatic_features
import create_label_sheets as sheets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "sample_vid"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
WINDOW_LABEL_PATH = LABEL_DIR / "manual_window_labels.csv"
VIDEO_LABEL_PATH = LABEL_DIR / "manual_video_labels.csv"

VIDEO_CANVAS_SIZE = (720, 405)
DEFAULT_WINDOW_SIZE = 3.0
DEFAULT_STEP_SIZE = 3.0
VALID_LABEL_VALUES = {"Y", "N"}
LABELLED_VALUE = "Y"
LOOK_AWAY_LEVEL_OPTIONS = [
    ("0", "0 — No clear look-away"),
    ("1", "1 — One brief look-away (about 0.3–1.5s)"),
    ("2", "2 — Sustained (>1.5s total) or repeated (≥2 times)"),
]


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
    level = row.get(sheets.LOOK_AWAY_LEVEL_COLUMN, "")
    quality = row.get("annotation_quality", "")
    return (
        level in sheets.TRAINABLE_LOOK_AWAY_LEVEL_VALUES and quality == "clear"
    ) or (
        level in {"", "U"} and quality in {"unclear", "invalid"}
    )


def is_labelled(row: dict[str, str]) -> bool:
    return has_valid_scores(row)


def label_progress(filenames: list[str]) -> tuple[int, int, float]:
    filename_set = {label_filename_for(filename) for filename in filenames}
    rows = [
        row
        for row in read_csv_rows(WINDOW_LABEL_PATH)
        if row.get("filename") in filename_set
    ]
    completed = sum(1 for row in rows if is_labelled(row))
    total = len(rows)
    percent = (completed / total * 100) if total else 0.0
    return completed, total, percent


def backfill_labelled_flags(filenames: list[str]) -> None:
    filename_set = {label_filename_for(filename) for filename in filenames}
    rows = read_csv_rows(WINDOW_LABEL_PATH)
    changed = False
    for row in rows:
        if row.get("filename") not in filename_set:
            continue
        if not row.get("is_labeled") and has_valid_scores(row):
            row["is_labeled"] = LABELLED_VALUE
            changed = True

    if changed:
        write_csv_rows(WINDOW_LABEL_PATH, rows, sheets.WINDOW_LABEL_COLUMNS)


def unlabelled_rows(filenames: list[str]) -> list[dict[str, str]]:
    filename_set = {label_filename_for(filename) for filename in filenames}
    return [
        row
        for row in read_csv_rows(WINDOW_LABEL_PATH)
        if row.get("filename") in filename_set and not is_labelled(row)
    ]


def legacy_look_away_rows(filenames: list[str]) -> list[dict[str, str]]:
    filename_set = {label_filename_for(filename) for filename in filenames}
    return [
        row
        for row in read_csv_rows(WINDOW_LABEL_PATH)
        if row.get("filename") in filename_set
        and (
            row.get("looking_away", "") in VALID_LABEL_VALUES
            or row.get("frequent_looking_away", "") in VALID_LABEL_VALUES
        )
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
        relabel_legacy_look_away: bool,
    ):
        self.root = root
        self.window_size = window_size
        self.step_size = step_size
        self.sample_limit = sample_limit
        self.relabel_legacy_look_away = relabel_legacy_look_away
        self.filenames = filenames
        self.filename = filenames[0] if filenames else ""
        self.rows: list[dict[str, str]] = []
        self.all_rows: list[dict[str, str]] = []
        self.index = 0
        self.cap: cv2.VideoCapture | None = None
        self.playback_end_frame: int | None = None
        self.after_id: str | None = None
        self.frame_image = None

        self.look_away_level_var = tk.StringVar(value="")
        self.quality_var = tk.StringVar(value="clear")
        self.notes_text: tk.Text | None = None

        self.build_ui()
        self.load_random_queue()
        self.load_current_window()

    def load_random_queue(self) -> None:
        for filename in self.filenames:
            ensure_label_rows(filename, self.window_size, self.step_size)
        self.all_rows = read_csv_rows(WINDOW_LABEL_PATH)
        if self.relabel_legacy_look_away:
            self.rows = legacy_look_away_rows(self.filenames)
        else:
            self.rows = unlabelled_rows(self.filenames)
        random.shuffle(self.rows)
        if self.sample_limit is not None:
            self.rows = self.rows[:self.sample_limit]
        self.index = 0
        scope = self.filename if len(self.filenames) == 1 else f"{len(self.filenames)} videos"
        self.root.title(f"Look-away Labeling - Random from {scope}")

    def build_ui(self) -> None:
        self.root.title(f"Look-away Labeling - {self.filename}")
        self.root.geometry("1180x640")
        self.root.minsize(980, 640)

        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        title = ttk.Label(outer, text="Look-away Window Labeling", font=("Arial", 20, "bold"))
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
        controls.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(controls, text="Previous Window", command=self.previous_window).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(controls, text="Replay", command=self.replay).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6,
        )
        ttk.Button(controls, text="Skip", command=self.skip).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=(6, 0),
        )

        form = ttk.Frame(outer)
        form.grid(row=1, column=1, sticky="nsew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Task").grid(row=0, column=0, sticky="w", pady=(0, 12))
        ttk.Label(
            form,
            text="Judge look-away only; do not label other behaviours in this pass.",
            wraplength=360,
            foreground="#56657a",
        ).grid(row=0, column=1, sticky="w", pady=(0, 12))

        ttk.Label(form, text="Quality").grid(row=1, column=0, sticky="w", pady=(0, 12))
        ttk.Combobox(
            form,
            textvariable=self.quality_var,
            values=("clear", "unclear", "invalid"),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=(0, 12))

        row_index = 2
        ttk.Label(form, text="Look-away level").grid(
            row=row_index, column=0, sticky="nw", pady=4
        )
        level_frame = ttk.Frame(form)
        level_frame.grid(row=row_index, column=1, sticky="ew", pady=4)
        ttk.Label(
            level_frame,
            text=(
                "Ignore movements shorter than about 0.3s. Use 0 for no clear look-away; "
                "1 for one brief look-away lasting about 0.3–1.5s; 2 when look-away lasts "
                "more than 1.5s in total or occurs at least twice. For clips that cannot "
                "be judged, choose quality unclear or invalid without selecting a level."
            ),
            wraplength=360,
            foreground="#56657a",
        ).grid(row=0, column=0, sticky="w")
        for option_index, (value, label) in enumerate(LOOK_AWAY_LEVEL_OPTIONS, 1):
            ttk.Radiobutton(
                level_frame,
                text=label,
                value=value,
                variable=self.look_away_level_var,
            ).grid(row=option_index, column=0, sticky="w", pady=(2, 0))
        row_index += 1

        ttk.Label(form, text="Notes").grid(row=row_index, column=0, sticky="nw", pady=(12, 4))
        self.notes_text = tk.Text(form, height=4, wrap="word")
        self.notes_text.bind("<Return>", self.submit_from_key)
        self.notes_text.bind("<Shift-Return>", self.insert_notes_newline)
        self.notes_text.grid(row=row_index, column=1, sticky="ew", pady=(12, 4))
        row_index += 1

        self.status_label = ttk.Label(form, text="", foreground="#3c475c")
        self.status_label.grid(row=row_index, column=0, columnspan=2, sticky="w", pady=(8, 8))
        row_index += 1

        ttk.Button(form, text="Label & Next", command=self.submit).grid(
            row=row_index,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(4, 0),
        )

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Return>", self.submit_from_key)
        self.root.bind("<Shift-Return>", self.insert_notes_newline)
        self.install_space_replay_bindings()

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

        self.look_away_level_var.set(
            row.get(sheets.LOOK_AWAY_LEVEL_COLUMN, "")
        )
        self.quality_var.set(row.get("annotation_quality") or "clear")

        assert self.notes_text is not None
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert("1.0", row.get("notes", ""))

        self.status_label.config(
            text="Choose quality and look-away level, then press Enter."
        )
        self.replay()

    def install_space_replay_bindings(self) -> None:
        replay_tag = "ReplayOnlySpace"
        self.root.bind_class(replay_tag, "<KeyPress-space>", self.replay_from_key)
        self.root.bind_class(replay_tag, "<KeyRelease-space>", self.block_key)
        self.prepend_bindtag(self.root, replay_tag)

    def prepend_bindtag(self, widget: tk.Widget, tag: str) -> None:
        bindtags = widget.bindtags()
        if tag not in bindtags:
            widget.bindtags((tag, *bindtags))
        for child in widget.winfo_children():
            self.prepend_bindtag(child, tag)

    def replay_from_key(self, event=None) -> str:
        self.replay()
        return "break"

    def block_key(self, event=None) -> str:
        return "break"

    def submit_from_key(self, event=None) -> str:
        self.submit()
        return "break"

    def insert_notes_newline(self, event=None) -> str:
        if self.notes_text is not None and self.root.focus_get() is self.notes_text:
            self.notes_text.insert(tk.INSERT, "\n")
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

        level = self.look_away_level_var.get()
        quality = self.quality_var.get()
        if quality == "clear":
            if level not in sheets.TRAINABLE_LOOK_AWAY_LEVEL_VALUES:
                self.status_label.config(text="Choose look-away level 0, 1, or 2 before saving.")
                return
            saved_level = level
        else:
            saved_level = ""

        row = self.rows[self.index]
        target_key = row_key(row)
        notes = self.notes_text.get("1.0", tk.END).strip() if self.notes_text else ""

        for all_row in self.all_rows:
            if row_key(all_row) != target_key:
                continue

            all_row[sheets.LOOK_AWAY_LEVEL_COLUMN] = saved_level
            row[sheets.LOOK_AWAY_LEVEL_COLUMN] = saved_level
            all_row["is_labeled"] = LABELLED_VALUE
            all_row["annotation_quality"] = self.quality_var.get()
            all_row["notes"] = notes
            row["is_labeled"] = all_row["is_labeled"]
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


def ensure_automatic_feature_rows(
    filenames: list[str],
    frame_stride: int,
    force: bool,
) -> list[str]:
    ready_filenames = []
    for index, filename in enumerate(filenames, 1):
        video_path = source_video_path(filename)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        label_filename = label_filename_for(filename)
        window_csv_path = OUTPUT_DIR / f"{Path(label_filename).stem}_windows.csv"
        if window_csv_path.exists() and not force:
            print(
                f"[{index}/{len(filenames)}] Automatic features already exist: "
                f"{window_csv_path.relative_to(PROJECT_ROOT)}"
            )
            ready_filenames.append(filename)
            continue

        print(
            f"[{index}/{len(filenames)}] Running automatic analysis for "
            f"{video_path.name} -> {label_filename}",
            flush=True,
        )
        try:
            summary_path, windows_path = automatic_features.analyse_video(video_path, frame_stride)
        except Exception as exc:
            print(f"  Automatic analysis failed for {video_path.name}: {exc}", file=sys.stderr)
            if window_csv_path.exists():
                print(f"  Keeping existing features: {window_csv_path.relative_to(PROJECT_ROOT)}")
                ready_filenames.append(filename)
            else:
                print(f"  Skipping manual labelling for {video_path.name}; no automatic window features exist.")
            continue

        print(f"  Wrote {summary_path.relative_to(PROJECT_ROOT)}")
        print(f"  Wrote {windows_path.relative_to(PROJECT_ROOT)}")
        ready_filenames.append(filename)

    return ready_filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the desktop window labelling tool.")
    parser.add_argument("--filename", help="Video filename to label, for example sample1.mp4.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--limit", type=int, help="Maximum number of random windows to label in this session.")
    parser.add_argument(
        "--skip-auto-analysis",
        action="store_true",
        help="Open manual labelling without first generating automatic window features.",
    )
    parser.add_argument(
        "--force-auto-analysis",
        action="store_true",
        help="Regenerate automatic window feature CSV files before manual labelling.",
    )
    parser.add_argument(
        "--analysis-frame-stride",
        type=int,
        default=10,
        help="Analyse one frame every N frames during automatic feature extraction.",
    )
    parser.add_argument(
        "--relabel-legacy-look-away",
        action="store_true",
        help=(
            "Review only windows that have an old Y/N looking_away label and "
            "replace migrated legacy labels with an explicit 0/1/2 level or non-clear quality."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        filenames = choose_filenames(args.filename)
        if args.window_size <= 0 or args.step_size <= 0:
            raise RuntimeError("--window-size and --step-size must be positive.")
        if args.limit is not None and args.limit <= 0:
            raise RuntimeError("--limit must be positive when provided.")
        if args.analysis_frame_stride < 1:
            raise RuntimeError("--analysis-frame-stride must be >= 1.")
        if not args.skip_auto_analysis:
            if args.window_size != DEFAULT_WINDOW_SIZE or args.step_size != DEFAULT_STEP_SIZE:
                raise RuntimeError(
                    "Automatic feature extraction currently uses 3 second windows with a 3 second step. "
                    "Use the default labelling window settings, or pass --skip-auto-analysis."
                )
            filenames = ensure_automatic_feature_rows(
                filenames,
                frame_stride=args.analysis_frame_stride,
                force=args.force_auto_analysis,
            )
            if not filenames:
                raise RuntimeError("No videos have automatic window features available for manual labelling.")
        for filename in filenames:
            ensure_label_rows(filename, args.window_size, args.step_size)
        backfill_labelled_flags(filenames)

        rows = (
            legacy_look_away_rows(filenames)
            if args.relabel_legacy_look_away
            else unlabelled_rows(filenames)
        )
        if not rows:
            queue_name = "legacy look-away" if args.relabel_legacy_look_away else "unlabelled"
            raise RuntimeError(f"No {queue_name} window rows found.")

        scope = filenames[0] if len(filenames) == 1 else f"{len(filenames)} videos"
        session_count = min(len(rows), args.limit) if args.limit is not None else len(rows)
        starting_completed, _, _ = label_progress(filenames)
        print(f"Prepared {session_count} random windows from {len(rows)} unlabelled windows in {scope}.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    root = tk.Tk()
    labeler = WindowLabeler(
        root,
        filenames,
        args.window_size,
        args.step_size,
        args.limit,
        args.relabel_legacy_look_away,
    )
    root.mainloop()

    completed, total, percent = label_progress(filenames)
    session_completed = (
        min(labeler.index, session_count)
        if args.relabel_legacy_look_away
        else max(0, completed - starting_completed)
    )
    print(
        "Labeling progress: "
        f"{completed}/{total} clips completed ({percent:.2f}%). "
        f"This session: {session_completed}/{session_count}."
    )


if __name__ == "__main__":
    main()
