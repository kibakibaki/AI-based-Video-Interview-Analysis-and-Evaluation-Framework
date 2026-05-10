from dataclasses import dataclass


@dataclass
class FrameObservation:
    face_detected: bool
    looking_at_camera: bool


class ConfidenceScorer:
    """
    Tracks confidence-related rubric scores over a camera/video stream.

    Current rubric:
        eye_contact: percentage of detected-face frames where the candidate is
        looking at the camera.

    Future rubrics can be added to _rubric_scores and _rubric_weights without
    changing the gaze detection code.
    """

    def __init__(self):
        self.total_frames = 0
        self.face_detected_frames = 0
        self.looking_at_camera_frames = 0
        self._rubric_scores = {
            "eye_contact": 0.0,
        }
        self._rubric_weights = {
            "eye_contact": 1.0,
        }

    def update(self, observation):
        self.total_frames += 1

        if observation.face_detected:
            self.face_detected_frames += 1

        if observation.looking_at_camera:
            self.looking_at_camera_frames += 1

        self._rubric_scores["eye_contact"] = self.eye_contact_score

    @property
    def eye_contact_score(self):
        if self.face_detected_frames == 0:
            return 0.0

        return (self.looking_at_camera_frames / self.face_detected_frames) * 100

    @property
    def face_visibility_ratio(self):
        if self.total_frames == 0:
            return 0.0

        return self.face_detected_frames / self.total_frames

    @property
    def overall_score(self):
        total_weight = sum(self._rubric_weights.values())
        if total_weight <= 0:
            return 0.0

        weighted_score = 0.0
        for rubric, score in self._rubric_scores.items():
            weighted_score += score * self._rubric_weights.get(rubric, 0.0)

        return weighted_score / total_weight

    @property
    def label(self):
        score = self.overall_score

        if self.face_detected_frames == 0:
            return "No face detected"
        if score >= 75:
            return "High confidence"
        if score >= 45:
            return "Moderate confidence"
        return "Low confidence"

    def report(self):
        return {
            "overall_score": round(self.overall_score, 2),
            "label": self.label,
            "rubrics": {
                "eye_contact": {
                    "score": round(self.eye_contact_score, 2),
                    "weight": self._rubric_weights["eye_contact"],
                    "description": "Share of detected-face frames looking at the camera.",
                },
            },
            "metrics": {
                "total_frames": self.total_frames,
                "face_detected_frames": self.face_detected_frames,
                "looking_at_camera_frames": self.looking_at_camera_frames,
                "face_visibility_ratio": round(self.face_visibility_ratio, 3),
            },
        }


def print_confidence_report(report):
    print("\n Confidence score ")
    print(f"Overall: {report['overall_score']:.2f}/100 ({report['label']})")

    eye_contact = report["rubrics"]["eye_contact"]
    print(f"Eye contact: {eye_contact['score']:.2f}/100")

    metrics = report["metrics"]
    print(f"Frames analysed: {metrics['total_frames']}")
    print(f"Face detected frames: {metrics['face_detected_frames']}")
    print(f"Looking-at-camera frames: {metrics['looking_at_camera_frames']}")
    print(f"Face visibility ratio: {metrics['face_visibility_ratio']:.3f}")
