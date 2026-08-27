import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app as app_module


class AppDirectoryConfigurationTests(unittest.TestCase):
    def test_app_runtime_directories_are_separate_from_training_data(self):
        self.assertEqual(
            app_module.UPLOAD_DIR,
            app_module.PROJECT_DIR / "data" / "app_output" / "uploads",
        )
        self.assertEqual(
            app_module.OUTPUT_DIR,
            app_module.PROJECT_DIR / "data" / "app_output" / "analysis",
        )
        self.assertNotEqual(
            app_module.OUTPUT_DIR,
            app_module.PROJECT_DIR / "data" / "output",
        )


class AppDataIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.upload_dir = self.project_dir / "data" / "app_output" / "uploads"
        self.output_dir = self.project_dir / "data" / "app_output" / "analysis"
        self.training_output_dir = self.project_dir / "data" / "output"
        self.training_output_dir.mkdir(parents=True)
        self.training_sentinel = self.training_output_dir / "sample1.csv"
        self.training_sentinel.write_text("training data\n", encoding="utf-8")

        self.path_patches = (
            patch.object(app_module, "PROJECT_DIR", self.project_dir),
            patch.object(app_module, "UPLOAD_DIR", self.upload_dir),
            patch.object(app_module, "OUTPUT_DIR", self.output_dir),
        )
        for path_patch in self.path_patches:
            path_patch.start()

        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

    def tearDown(self):
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.temp_dir.cleanup()

    def test_upload_and_analysis_use_uuid_and_app_output_only(self):
        visual_report = {
            "features": {"eye_contact_ratio": 0.6},
            "window_features": [
                {"window_start": 0.0, "window_end": 3.0, "eye_contact_ratio": 0.6}
            ],
        }
        gaze_shift_report = {
            "task": "binary_gaze_shift",
            "model": "test-model.joblib",
            "summary": {
                "total_windows": 1,
                "observable_windows": 1,
                "classified_windows": 1,
                "no_gaze_shift_windows": 0,
                "gaze_shift_windows": 1,
                "uncertain_windows": 0,
                "unobservable_windows": 0,
                "gaze_shift_ratio": 1.0,
            },
            "windows": [
                {
                    "window_start": 0.0,
                    "window_end": 3.0,
                    "state": "gaze_shift",
                    "reason": None,
                    "face_visibility_ratio": 1.0,
                    "visual_observability_ratio": 1.0,
                    "prediction_probability": 0.8,
                    "probabilities": {"no_gaze_shift": 0.2, "gaze_shift": 0.8},
                }
            ],
        }

        with patch.object(app_module, "_get_video_duration", return_value=3.0):
            upload_response = self.client.post(
                "/upload",
                data={"video": (io.BytesIO(b"video bytes"), "candidate.mp4")},
                content_type="multipart/form-data",
            )

            self.assertEqual(upload_response.status_code, 200)
            upload_json = upload_response.get_json()
            filename = upload_json["filename"]
            parsed_id = UUID(Path(filename).stem)
            self.assertEqual(parsed_id.version, 4)
            self.assertEqual(Path(filename).suffix, ".mp4")
            self.assertEqual(
                upload_json["path"],
                f"data/app_output/uploads/{filename}",
            )
            self.assertTrue((self.upload_dir / filename).exists())

            with patch.object(
                app_module,
                "analyse_gaze",
                return_value=([(0.25, 0.75)], 0.5, visual_report),
            ):
                with patch.object(
                    app_module,
                    "classify_gaze_shift_video",
                    return_value=gaze_shift_report,
                ):
                    analysis_response = self.client.post(
                        "/analyse",
                        json={"filename": filename},
                    )

        self.assertEqual(analysis_response.status_code, 200)
        analysis_json = analysis_response.get_json()["analysis"]
        self.assertNotIn("confidence_report", analysis_json)
        self.assertEqual(
            analysis_json["gaze_shift_report"]["windows"][0]["state"],
            "gaze_shift",
        )
        stem = Path(filename).stem
        self.assertEqual(
            analysis_json["csv_path"],
            f"data/app_output/analysis/{stem}.csv",
        )
        self.assertEqual(
            analysis_json["window_csv_path"],
            f"data/app_output/analysis/{stem}_windows.csv",
        )
        self.assertTrue((self.output_dir / f"{stem}.csv").exists())
        self.assertTrue((self.output_dir / f"{stem}_windows.csv").exists())
        self.assertEqual(
            self.training_sentinel.read_text(encoding="utf-8"),
            "training data\n",
        )
        self.assertEqual(
            list(self.training_output_dir.iterdir()),
            [self.training_sentinel],
        )


if __name__ == "__main__":
    unittest.main()
