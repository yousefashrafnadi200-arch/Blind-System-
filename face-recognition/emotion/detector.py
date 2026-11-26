import os
from typing import Callable, Optional, Sequence


# Arabic labels for display. The model keeps the canonical English names so the
# output stays stable for logs, logic and future integration.
LABELS_AR = {
    "Anger": "غضبان",
    "Contempt": "مستخف",
    "Disgust": "مقرف",
    "Fear": "خايف",
    "Happiness": "فرحان",
    "Neutral": "طبيعي",
    "Sadness": "زعلان",
    "Surprise": "مندهش",
    "no face": "مفيش وش",
    "unknown": "مش معروف",
    "unavailable": "غير متاح",
}


def to_arabic(label: Optional[str]) -> str:
    """Translate an emotion label to Arabic (unknown labels pass through)."""

    if not label:
        return LABELS_AR["unknown"]

    return LABELS_AR.get(str(label), str(label))


class EmotionBackend:
    """Interface for a face-emotion backend."""

    def classify(self, frame) -> Optional[str]:
        raise NotImplementedError


class EmotionDetector:
    """Independent face-emotion detection module.

    Deliberately separate from the identity/face-recognition database so it can
    be built, tested and run on its own.
    """

    def __init__(
        self,
        emotions: Sequence[str] = (
            "Anger",
            "Contempt",
            "Disgust",
            "Fear",
            "Happiness",
            "Neutral",
            "Sadness",
            "Surprise",
        ),
        predictor: Optional[Callable[[object], Optional[str]]] = None,
        backend: Optional[EmotionBackend] = None,
        model_path: Optional[str] = None,
    ):
        self.emotions = list(emotions)
        self.model_path = model_path
        self.predictor = predictor
        self.backend = backend

    @property
    def is_configured(self) -> bool:
        return self.backend is not None or self.predictor is not None

    def detect(self, frame, arabic: bool = False) -> str:
        """Return the dominant emotion label for `frame`.

        With ``arabic=True`` the label is translated for display
        ("Happiness" -> "فرحان").
        """

        if self.backend is not None:
            label = self.backend.classify(frame)
        elif self.predictor is not None:
            label = self.predictor(frame)
        else:
            raise RuntimeError(
                "EmotionDetector is not configured. Pass a `predictor` or "
                "`backend`, or set `model_path`."
            )

        label = str(label or "unknown")
        return to_arabic(label) if arabic else label


class OnnxEmotionBackend(EmotionBackend):
    """Emotion backend using HSEmotion (EfficientNet-B0) via onnxruntime.

    The model was trained on AffectNet and predicts eight emotions from an
    aligned 224x224 RGB face.
    """

    LABELS = (
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    )

    # ImageNet statistics, as used during training.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path: str, det_size=(320, 320)):
        import cv2
        import numpy as np
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        self.cv2 = cv2
        self.np = np
        self.face_align = face_align

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean = np.array(self.MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.STD, dtype=np.float32).reshape(1, 1, 3)

        # Detection only: no recognition/landmark models are needed here.
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection"],
        )
        self.face_app.prepare(ctx_id=-1, det_size=det_size)

    def _largest_face(self, frame):
        faces = self.face_app.get(frame)

        if len(faces) == 0:
            return None

        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def _prepare(self, frame, face):
        cv2 = self.cv2
        np = self.np

        # Align the face using its 5 landmarks; the model expects aligned faces.
        aligned = self.face_align.norm_crop(
            frame,
            landmark=face.kps,
            image_size=224,
        )

        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std

        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def classify(self, frame) -> Optional[str]:
        np = self.np

        if frame is None:
            return None

        face = self._largest_face(frame)

        if face is None:
            return "no face"

        blob = self._prepare(frame, face)

        scores = self.session.run(
            [self.output_name],
            {self.input_name: blob},
        )[0][0]

        scores = np.asarray(scores, dtype=np.float32)
        exp = np.exp(scores - scores.max())
        probabilities = exp / exp.sum()

        return self.LABELS[int(np.argmax(probabilities))]
