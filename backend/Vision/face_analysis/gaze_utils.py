from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GazeObservation:
    direction: Optional[str]
    left_pupil: Optional[Tuple[int, int]]
    right_pupil: Optional[Tuple[int, int]]
    horizontal_ratio: Optional[float]
    vertical_ratio: Optional[float]
    is_blinking: bool

    @property
    def is_center(self):
        return self.direction == "center"


class PupilGazeTracker:
    """
    Thin adapter around antoinelame/GazeTracking.

    The third-party dependency is imported lazily so the rest of the backend can
    still load even before users install the optional gaze-tracking package.
    """

    def __init__(self):
        try:
            from gaze_tracking import GazeTracking
        except ImportError as exc:
            raise ImportError(
                "GazeTracking is not installed. Install dependencies with "
                "`pip install -r backend/requirements.txt`. If dlib fails on "
                "macOS, install CMake first with `brew install cmake`."
            ) from exc

        self._tracker = GazeTracking()

    def analyse_frame(self, frame):
        self._tracker.refresh(frame)

        direction = None
        if self._tracker.is_left():
            direction = "left"
        elif self._tracker.is_right():
            direction = "right"
        elif self._tracker.is_center():
            direction = "center"

        return GazeObservation(
            direction=direction,
            left_pupil=self._tracker.pupil_left_coords(),
            right_pupil=self._tracker.pupil_right_coords(),
            horizontal_ratio=self._safe_call(self._tracker.horizontal_ratio),
            vertical_ratio=self._safe_call(self._tracker.vertical_ratio),
            is_blinking=bool(self._tracker.is_blinking()),
        )

    @staticmethod
    def _safe_call(callback):
        try:
            return callback()
        except (TypeError, ZeroDivisionError):
            return None
