import os
from typing import Callable, Optional, Sequence


class EmotionBackend:
    """Interface for a face-emotion backend."""

    def classify(self, frame) -> Optional[str]:
        raise NotImplementedError


class EmotionDetector:
    """Independent face-emotion detection module.

    This is deliberately separate from the identity/face-recognition database so
    it can be built, tested and run on its own.

    A backend/classifier callable receives a frame (numpy array, BGR like OpenCV)
    and returns the emotion label (e.g. "happiness").
    """

    def __init__(
        self,
        emotions: Sequence[str] = (
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
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

    def detect(self, frame) -> str:
        """Return the dominant emotion label for `frame`."""

        if self.backend is not None:
            label = self.backend.classify(frame)
        elif self.predictor is not None:
            label = self.predictor(frame)
        else:
            raise RuntimeError(
                "EmotionDetector is not configured. Pass a `predictor` or "
                "`backend`, or set `model_path`."
            )

        return str(label or "unknown")


class OnnxEmotionBackend(EmotionBackend):
    """Emotion backend using the FER+ ONNX model with onnxruntime.

    FER+ predicts one of eight emotions from a 64x64 grayscale face crop.
    Faces are located with InsightFace's detection model, which is already
    cached locally by the face-recognition module.
    """

    LABELS = (
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    )

    def __init__(self, model_path: str, det_size=(320, 320)):
        import cv2
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        self.cv2 = cv2

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Detection only: no recognition/landmark models are needed here.
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection"],
        )
        self.face_app.prepare(ctx_id=-1, det_size=det_size)

    def _largest_face_box(self, frame):
        faces = self.face_app.get(frame)

        if len(faces) == 0:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = (int(value) for value in face.bbox)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, height)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def classify(self, frame) -> Optional[str]:
        import numpy as np

        if frame is None:
            return None

        box = self._largest_face_box(frame)

        if box is None:
            return "no face"

        x1, y1, x2, y2 = box

        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]

        face = self.cv2.resize(crop, (64, 64)).astype(np.float32)
        face = (face - 127.5) / 127.5
        blob = face.reshape(1, 1, 64, 64)

        scores = self.session.run(
            [self.output_name],
            {self.input_name: blob},
        )[0][0]

        scores = np.asarray(scores, dtype=np.float32)
        exp = np.exp(scores - scores.max())
        probabilities = exp / exp.sum()

        return self.LABELS[int(np.argmax(probabilities))]
