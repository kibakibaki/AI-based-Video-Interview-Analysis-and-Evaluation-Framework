"""Compatibility import for the shared binary gaze-shift feature extractor."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Vision.face_analysis.gaze_shift_features import (  # noqa: E402,F401
    FEATURE_NAMES,
    extract_frame_features,
    extract_landmark_features,
    missing_feature_vector,
)


__all__ = [
    "FEATURE_NAMES",
    "extract_frame_features",
    "extract_landmark_features",
    "missing_feature_vector",
]
