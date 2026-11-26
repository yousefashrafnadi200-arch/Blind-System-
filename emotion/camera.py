import threading

import cv2


class CameraStream:
    """Continuously grabs frames in a background thread.

    Kept inside the emotion module so the module has no dependency on the
    face-recognition package: copy it, change the index, and it runs anywhere.

    A plain ``camera.read()`` loop falls behind whenever processing is slow:
    the driver keeps queueing frames and ``read`` returns old ones, which is
    what makes the preview feel delayed. This class always keeps only the
    newest frame, so the preview stays live.
    """

    def __init__(
        self,
        index=0,
        width=640,
        height=480,
        backend=None,
        mirror=True,
    ):
        self.index = index
        self.mirror = mirror

        self.lock = threading.Lock()
        self.frame = None
        self._frame_count = 0

        self.running = False
        self.thread = None

        if backend is None:
            # DirectShow opens much faster than the default MSMF on Windows.
            backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)

        self.capture = cv2.VideoCapture(index, backend)

        if not self.capture.isOpened() and backend != cv2.CAP_ANY:
            self.capture.release()
            self.capture = cv2.VideoCapture(index)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep the driver's queue as short as possible.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self):
        return self.capture.isOpened()

    def start(self):

        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name="camera",
            daemon=True
        )
        self.thread.start()

    def read(self):
        """Return a copy of the newest frame, or None if there is none yet."""

        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    @property
    def frame_count(self):
        """Number of frames the camera has delivered so far."""

        with self.lock:
            return self._frame_count

    def release(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(2.0)
            self.thread = None

        self.capture.release()

    def _run(self):

        while self.running:

            ok, frame = self.capture.read()

            if not ok:
                continue

            if self.mirror:
                # Flip once, before anything else, so anything drawn later
                # lines up with what is on screen.
                frame = cv2.flip(frame, 1)

            with self.lock:
                self.frame = frame
                self._frame_count += 1
