import threading
import time


class FaceWorker:
    """Runs detection/recognition off the GUI thread.

    The main loop pushes frames with :meth:`update_frame`; the worker only
    processes frames it has not seen yet, at most ``max_fps`` times a second,
    and publishes results atomically.
    """

    def __init__(self, detector, database, max_fps=10.0):

        self.detector = detector
        self.database = database

        self.frame = None
        self.frame_id = 0
        self.results = []

        self.lock = threading.Lock()

        self.running = False
        self.thread = None
        self.error = None

        self.processed_frames = 0
        self.last_process_time = 0.0

        self.min_interval = (
            1.0 / max_fps
            if max_fps and max_fps > 0
            else 0.0
        )

    def update_frame(self, frame):

        with self.lock:
            self.frame = frame.copy()
            self.frame_id += 1

    def get_results(self):

        with self.lock:
            return self.results.copy()

    def start(self):

        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            name="face-worker",
            daemon=True
        )

        self.thread.start()

    def stop(self, timeout=5.0):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout)

            if self.thread.is_alive():
                print("Face worker did not stop in time.")

            self.thread = None

    def _run(self):

        last_frame_id = -1

        while self.running:

            with self.lock:
                frame_id = self.frame_id
                frame = (
                    self.frame.copy()
                    if self.frame is not None
                    else None
                )

            # Nothing new to do: yield instead of re-running inference.
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = frame_id

            started = time.time()

            try:
                results = self._process(frame)
            except Exception as exc:
                # A single bad frame must not kill the worker thread.
                self.error = exc
                print(f"Face worker error: {exc}")
                time.sleep(0.5)
                continue

            with self.lock:
                self.results = results
                self.processed_frames += 1
                self.last_process_time = time.time() - started

            # Throttle: keep the CPU free for the preview.
            remaining = self.min_interval - (time.time() - started)

            if remaining > 0:
                time.sleep(remaining)

    def _process(self, frame):

        results = []

        for face in self.detector.detect(frame):

            x1, y1, x2, y2 = (int(value) for value in face.bbox)

            name, similarity = self.database.recognize(face)

            results.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    name,
                    similarity
                )
            )

        return results
