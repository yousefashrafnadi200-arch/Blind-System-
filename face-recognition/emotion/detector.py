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
    and returns the emotion label (e.g. "happy").
    """

    def __init__(
        self,
        emotions: Sequence[str] = (
            "neutral",
            "happy",
            "sad",
            "angry",
            "surprised",
            "fear",
            "disgust",
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


class MediaPipeEmotionBackend(EmotionBackend):
    """Emotion backend powered by Google MediaPipe emotion recognition."""

    def __init__(self, model_asset_path: Optional[str] = None):
        self.model_asset_path = model_asset_path
        self.error: Optional[str] = None
        self.recognizer = None

        if model_asset_path is None:
            self.error = "MediaPipeEmotionBackend requires a model_asset_path."
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision

            base_options = mp.tasks.BaseOptions(
                model_asset_path=model_asset_path,
                delegate="CPU",
            )

            # Prefer the newer task-based emotion recognizer when available.
            if hasattr(vision, "EffortEmotionRecognizer"):
                options = vision.EffortEmotionRecognizerOptions(
                    model_asset_path=model_asset_path,
                    delegate="CPU",
                )
                self.recognizer = vision.EffortEmotionRecognizer.create_from_options(
                    options=options,
                    base_options=base_options,
                )
            elif hasattr(vision, "FaceEmotionRecognizer"):
                options = vision.FaceEmotionRecognizerOptions(
                    model_asset_path=model_asset_path,
                    delegate="CPU",
                )
                self.recognizer = vision.FaceEmotionRecognizer.create_from_options(
                    options=options,
                    base_options=base_options,
                )
            else:
                raise ImportError(
                    "No emotion recognizer found in installed mediapipe version."
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.error = str(exc)

    def classify(self, frame) -> Optional[str]:
        if self.recognizer is None:
            raise RuntimeError(
                f"MediaPipe emotion backend is not usable: {self.error}"
            )

        import cv2

        if frame is None:
            return None

        result = self.recognizer.recognize(frame)

        emotion = getattr(result, "emotion_category", None) or getattr(
            result, "emotion", None
        )

        if emotion is None:
            return None

        # emotion may be a single value or a list-like of values.
        try:
            emotions = list(emotion)
        except TypeError:
            emotions = [emotion]

        if not emotions:
            return None

        return str(emotions[0])
