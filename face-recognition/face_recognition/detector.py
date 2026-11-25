from insightface.app import FaceAnalysis


class FaceDetector:
    """Thin wrapper around InsightFace's ``FaceAnalysis``.

    Only the detection and recognition models are loaded: the landmark and
    gender/age models cost a lot of time per frame and are not needed here.
    """

    def __init__(
        self,
        model_name="buffalo_l",
        det_size=(320, 320),
        providers=None,
        ctx_id=-1,
        allowed_modules=None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if allowed_modules is None:
            allowed_modules = ["detection", "recognition"]

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=allowed_modules
        )

        # ctx_id=-1 means CPU; 0+ would select that GPU index.
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=det_size
        )

    def detect(self, frame):
        """
        Detect faces and extract their embeddings.

        Args:
            frame: BGR image, as returned by ``cv2.VideoCapture.read``.

        Returns:
            list of InsightFace Face objects.
        """

        # OpenCV uses BGR, which is what InsightFace expects.
        return self.app.get(frame)
