from typing import Callable, Optional, Sequence


class EmotionDetector:
    """Independent face-emotion detection module.

    This is deliberately separate from the identity/face-recognition database so
    it can be built, tested and run on its own.

    A `predictor` callable receives a frame (numpy array, BGR like OpenCV) and
    returns the emotion label (e.g. "happy"). You can plug in any emotion model
    that produces a label for a face, or a full pipeline that crops the face from
    the bbox first.
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
        model_path: Optional[str] = None,
    ):
        self.emotions = list(emotions)
        self.model_path = model_path
        self.predictor = predictor

    @property
    def is_configured(self) -> bool:
        return self.predictor is not None

    def detect(self, frame) -> str:
        """Return the dominant emotion label for `frame`."""

        if not self.is_configured:
            raise RuntimeError(
                "EmotionDetector is not configured. Pass a `predictor` "
                "or set `model_path`."
            )

        label = self.predictor(frame) if self.predictor is not None else None
        return str(label or "unknown")
