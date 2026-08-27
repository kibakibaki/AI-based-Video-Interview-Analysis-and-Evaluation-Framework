import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "vision_events_v2"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import extract_sequences  # noqa: E402


class FrozenBinaryTaskTests(unittest.TestCase):
    def test_loader_has_no_three_class_target_switch(self):
        parameters = inspect.signature(extract_sequences.load_label_rows).parameters

        self.assertEqual(list(parameters), ["path"])

    def test_clear_levels_map_to_binary_and_quality_outcomes_are_excluded(self):
        fieldnames = [
            "clip_id",
            "filename",
            "window_start",
            "window_end",
            "look_away_level",
            "annotation_quality",
        ]
        rows = [
            ("w0", "sample1.mp4", "0", "3", "0", "clear"),
            ("w1", "sample1.mp4", "3", "6", "1", "clear"),
            ("w2", "sample1.mp4", "6", "9", "2", "clear"),
            ("wu", "sample1.mp4", "9", "12", "", "uncertain"),
            ("wo", "sample1.mp4", "12", "15", "", "unobservable"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(fieldnames)
                writer.writerows(rows)

            loaded = extract_sequences.load_label_rows(path)

        self.assertEqual([row["target"] for row in loaded], [0, 1, 1])
        self.assertEqual([row["level"] for row in loaded], [0, 1, 2])

    def test_repository_labels_use_frozen_quality_vocabulary(self):
        with (PROJECT_ROOT / "data" / "labels" / "manual_window_labels.csv").open(
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            qualities = {
                row["annotation_quality"]
                for row in csv.DictReader(csv_file)
                if row["annotation_quality"]
            }

        self.assertLessEqual(qualities, {"clear", "uncertain", "unobservable"})


if __name__ == "__main__":
    unittest.main()
