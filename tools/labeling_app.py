#!/usr/bin/env python3
"""
Desktop window labelling tool for sliding-window interview clips.

Run from the project root:

    python tools/labeling_app.py

Optional:

    python tools/labeling_app.py --filename sample1.mov
"""

from __future__ import annotations

import argparse
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
CLIP_ROOT = PROJECT_ROOT / "data" / "window_clips"
WINDOW_LABEL_PATH = LABEL_DIR / "manual_window_labels.csv"
VIDEO_LABEL_PATH = LABEL_DIR / "manual_video_labels.csv"

VIDEO_CANVAS_SIZE = (720, 405)
DEFAULT_WINDOW_SIZE = 5.0
DEFAULT_STEP_SIZE = 5.0
VALID_SCORES = {"1", "2", "3", "4", "5"}


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
    video_paths = sorted(VIDEO_DIR.iterdir(), key=sample_sort_key) if VIDEO_DIR.exists() else []
    for path in video_paths:
        if path.suffix.lower() in {".mp4", ".mov"}:
            filenames.append(path.name)

    for row in read_csv_rows(WINDOW_LABEL_PATH):
        filename = row.get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)

    return sorted(filenames, key=sample_sort_key)


def source_video_path(filename: str) -> Path:
    return VIDEO_DIR / filename


def rows_for_filename(filename: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv_rows(WINDOW_LABEL_PATH)
        if row.get("filename") == filename
    ]


def has_valid_scores(row: dict[str, str]) -> bool:
    return all(row.get(field["name"], "") in VALID_SCORES for field in sheets.LABEL_FIELDS)


def first_unlabelled_index(rows: list[dict[str, str]], start: int = 0) -> int:
    for index in range(start, len(rows)):
        if not has_valid_scores(rows[index]):
            return index
    return len(rows)


def first_incomplete_filename(filenames: list[str]) -> str:
    for filename in filenames:
        rows = rows_for_filename(filename)
        if not rows or first_unlabelled_index(rows) < len(rows):
            return filename
    return filenames[-1]


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
    rows = []
    window_start = 0.0

    while window_start < duration:
        window_end = min(window_start + window_size, duration)
        if window_end <= window_start:
            break

        clip_id = sheets.clip_id_for(filename, f"{window_start:.2f}", f"{window_end:.2f}")
        clip_path = CLIP_ROOT / Path(filename).stem / f"{clip_id}.mp4"
        window_duration = window_end - window_start
        row = {
            "clip_id": clip_id,
            "filename": filename,
            "source_csv": "",
            "clip_path": sheets.relative_path(clip_path),
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
        row.get("filename", ""),
        row.get("window_start", ""),
        row.get("window_end", ""),
    )


def ensure_label_rows(filename: str, window_size: float, step_size: float) -> list[dict[str, str]]:
    existing_rows = [sheets.normalise_existing_row(row) for row in read_csv_rows(WINDOW_LABEL_PATH)]
    existing_by_key = {row_key(row): row for row in existing_rows}
    generated_rows = build_dataset_rows_for_video(filename, window_size, step_size)

    merged_rows = [row for row in existing_rows if row.get("filename") != filename]
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
    rows = [sheets.normalise_existing_row(row) for row in read_csv_rows(VIDEO_LABEL_PATH)]
    existing = None
    remaining_rows = []
    for row in rows:
        if row.get("filename") == filename:
            existing = row
        else:
            remaining_rows.append(row)

    row = {
        "filename": filename,
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


def clip_can_open(clip_path: Path) -> bool:
    if not clip_path.exists() or clip_path.stat().st_size <= 1024:
        return False

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return False

    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def export_window_clips(filename: str, rows: list[dict[str, str]]) -> int:
    video_path = source_video_path(filename)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    created_count = 0

    try:
        for row in rows:
            clip_path = PROJECT_ROOT / row["clip_path"]
            if clip_can_open(clip_path):
                continue

            clip_path.parent.mkdir(parents=True, exist_ok=True)
            start = float(row["window_start"])
            end = float(row["window_end"])
            start_frame = max(0, int(round(start * fps)))
            end_frame = max(start_frame + 1, int(round(end * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            temp_path = clip_path.with_suffix(".tmp.mp4")
            temp_path.unlink(missing_ok=True)

            writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create clip file: {temp_path}")

            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)

            writer.release()

            if not clip_can_open(temp_path):
                temp_path.unlink(missing_ok=True)
                clip_path.unlink(missing_ok=True)
                raise RuntimeError(f"Generated clip is not readable: {clip_path}")

            clip_path.unlink(missing_ok=True)
            temp_path.rename(clip_path)
            created_count += 1
    finally:
        cap.release()

    return created_count


def ensure_current_clip(row: dict[str, str]) -> bool:
    clip_path = PROJECT_ROOT / row["clip_path"]
    if clip_can_open(clip_path):
        return True

    export_window_clips(row["filename"], [row])
    return clip_can_open(clip_path)


class WindowLabeler:
    def __init__(self, root: tk.Tk, filename: str, window_size: float, step_size: float):
        self.root = root
        self.window_size = window_size
        self.step_size = step_size
        self.filename = filename
        self.rows: list[dict[str, str]] = []
        self.all_rows: list[dict[str, str]] = []
        self.index = 0
        self.cap: cv2.VideoCapture | None = None
        self.after_id: str | None = None
        self.frame_image = None

        self.score_vars = {field["name"]: tk.StringVar() for field in sheets.LABEL_FIELDS}
        self.quality_var = tk.StringVar(value="clear")
        self.notes_text: tk.Text | None = None

        self.build_ui()
        self.load_video(filename)
        self.load_current_window()

    def load_video(self, filename: str) -> None:
        self.filename = filename
        ensure_label_rows(filename, self.window_size, self.step_size)
        self.rows = rows_for_filename(filename)
        self.all_rows = read_csv_rows(WINDOW_LABEL_PATH)
        export_window_clips(filename, self.rows)
        self.index = self.next_unlabelled_index(0)
        self.root.title(f"Window Labeling - {self.filename}")

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
            ttk.Spinbox(
                field_frame,
                from_=1,
                to=5,
                width=5,
                textvariable=self.score_vars[field["name"]],
                validate="key",
                validatecommand=(self.root.register(self.validate_score), "%P"),
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

    @staticmethod
    def validate_score(value: str) -> bool:
        return value == "" or value in VALID_SCORES

    def next_unlabelled_index(self, start: int) -> int:
        return first_unlabelled_index(self.rows, start)

    def load_next_incomplete_video(self) -> bool:
        filenames = available_filenames()
        try:
            current_position = filenames.index(self.filename)
        except ValueError:
            current_position = -1

        ordered_candidates = filenames[current_position + 1:] + filenames[:current_position + 1]
        for filename in ordered_candidates:
            ensure_label_rows(filename, self.window_size, self.step_size)
            rows = rows_for_filename(filename)
            if rows and first_unlabelled_index(rows) < len(rows):
                self.load_video(filename)
                return True
        return False

    def load_current_window(self) -> None:
        self.stop_video()

        if self.index >= len(self.rows):
            if self.load_next_incomplete_video():
                self.load_current_window()
                return
            self.progress_label.config(text="All videos completed")
            self.time_label.config(text="All windows completed")
            self.status_label.config(text="All sample_vid windows are labelled.")
            self.video_label.config(image="", text="All windows completed", fg="white")
            return

        row = self.rows[self.index]
        self.progress_label.config(text=f"{self.filename} | Window {self.index + 1} / {len(self.rows)}")
        self.time_label.config(text=f"{row['window_start']}s - {row['window_end']}s")

        for field in sheets.LABEL_FIELDS:
            self.score_vars[field["name"]].set(row.get(field["name"], ""))
        self.quality_var.set(row.get("annotation_quality") or "clear")

        assert self.notes_text is not None
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert("1.0", row.get("notes", ""))

        self.status_label.config(text="Score 1-5. Space = replay current window.")
        self.replay()

    def replay_from_key(self, event=None) -> str:
        self.replay()
        return "break"

    def replay(self) -> None:
        self.stop_video()
        if self.index >= len(self.rows):
            return

        row = self.rows[self.index]
        clip_path = PROJECT_ROOT / row["clip_path"]
        try:
            ensure_current_clip(row)
        except Exception as exc:
            messagebox.showerror("Video Error", f"Cannot prepare clip:\n{clip_path}\n\n{exc}")
            return

        self.cap = cv2.VideoCapture(str(clip_path))
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Cannot open clip:\n{clip_path}")
            return

        self.show_next_frame()

    def show_next_frame(self) -> None:
        if self.cap is None:
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

        scores = {}
        for field in sheets.LABEL_FIELDS:
            value = self.score_vars[field["name"]].get()
            if value not in VALID_SCORES:
                messagebox.showwarning("Missing Score", f"Please fill {field['label']} with 1-5.")
                return
            scores[field["name"]] = value

        row = self.rows[self.index]
        target_key = row_key(row)
        notes = self.notes_text.get("1.0", tk.END).strip() if self.notes_text else ""

        for all_row in self.all_rows:
            if row_key(all_row) != target_key:
                continue

            for field_name, value in scores.items():
                all_row[field_name] = value
                row[field_name] = value
            all_row["annotation_quality"] = self.quality_var.get()
            all_row["notes"] = notes
            row["annotation_quality"] = self.quality_var.get()
            row["notes"] = notes
            break

        write_csv_rows(WINDOW_LABEL_PATH, self.all_rows, sheets.WINDOW_LABEL_COLUMNS)
        self.status_label.config(text=f"Saved window {self.index + 1}.")
        self.index = self.next_unlabelled_index(self.index + 1)
        self.load_current_window()

    def close(self) -> None:
        self.stop_video()
        self.root.destroy()


def choose_filename(args_filename: str | None) -> str:
    filenames = available_filenames()
    if not filenames:
        raise RuntimeError("No videos found in sample_vid. Add a video first.")

    if args_filename:
        if args_filename not in filenames:
            raise RuntimeError(f"{args_filename} was not found in sample_vid.")
        return args_filename

    return first_incomplete_filename(filenames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the desktop window labelling tool.")
    parser.add_argument("--filename", help="Video filename to label, for example sample1.mov.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        filename = choose_filename(args.filename)
        if args.window_size <= 0 or args.step_size <= 0:
            raise RuntimeError("--window-size and --step-size must be positive.")
        ensure_label_rows(filename, args.window_size, args.step_size)
        rows = rows_for_filename(filename)
        if not rows:
            raise RuntimeError(f"No window rows found for {filename}.")

        created_count = export_window_clips(filename, rows)
        print(f"Prepared {len(rows)} windows for {filename}. Created {created_count} new clips.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    root = tk.Tk()
    WindowLabeler(root, filename, args.window_size, args.step_size)
    root.mainloop()


if __name__ == "__main__":
    main()
