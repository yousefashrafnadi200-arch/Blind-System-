import os
import time

import cv2

from camera import CameraStream
from face_recognition.detector import FaceDetector
from face_recognition.database import FaceDatabase
from face_recognition.worker import FaceWorker


# ---------------------------------------------------------------- settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# True = selfie view (what you see in a mirror). Set False for the raw
# camera orientation.
MIRROR = True

# Detection resolution: (640, 640) is more accurate, (320, 320) is much
# faster on CPU.
DETECT_SIZE = (320, 320)

# How many recognition passes per second. The preview stays at camera speed;
# only the boxes update at this rate.
MAX_FPS = 10.0

# Cosine similarity needed to accept a match.
THRESHOLD = 0.5
# ---------------------------------------------------------------- settings


def draw_results(frame, results):

    for (x1, y1, x2, y2, name, similarity) in results:

        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{name} ({similarity:.2f})"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        top = max(y1 - text_height - baseline - 6, 0)

        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + text_width + 6, top + text_height + baseline + 6),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (x1 + 3, top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )


def main():

    print("=" * 50)
    print("Face Recognition System")
    print("=" * 50)

    print("\nLoading InsightFace...")

    detector = FaceDetector(det_size=DETECT_SIZE)

    print("InsightFace loaded.")

    faces_folder = os.path.join(
        os.path.dirname(__file__),
        "Faces"
    )

    database = FaceDatabase(
        faces_folder,
        detector,
        threshold=THRESHOLD
    )

    database.load()

    print("\nStarting camera...")

    camera = CameraStream(
        index=CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        mirror=MIRROR
    )

    if not camera.is_opened():
        print("Failed to open camera.")
        return

    camera.start()

    worker = FaceWorker(
        detector,
        database,
        max_fps=MAX_FPS
    )

    worker.start()

    window_name = "Face Recognition"

    fps = 0.0
    fps_frames = 0
    fps_started = time.time()
    last_camera_count = 0

    detect_fps = 0.0
    detect_frames = 0
    detect_started = time.time()
    detect_latency = 0.0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                time.sleep(0.005)
                continue

            # Send latest frame to the AI worker
            worker.update_frame(frame)

            # Get latest recognition results
            results = worker.get_results()

            draw_results(frame, results)

            # Preview FPS: count only frames the camera actually delivered.
            camera_count = camera.frame_count

            if camera_count != last_camera_count:
                last_camera_count = camera_count
                fps_frames += 1
            elapsed = time.time() - fps_started

            if elapsed >= 0.5:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_started = time.time()

            # Periodic stats in the console, so slowness is measurable.
            detect_elapsed = time.time() - detect_started

            if detect_elapsed >= 2.0:
                processed = worker.processed_frames
                detect_fps = (processed - detect_frames) / detect_elapsed
                detect_frames = processed
                detect_started = time.time()
                detect_latency = worker.last_process_time

                print(
                    f"preview {fps:5.1f} fps | "
                    f"detect {detect_fps:4.1f} fps | "
                    f"inference {detect_latency * 1000:4.0f} ms"
                )

            cv2.putText(
                frame,
                f"{fps:.0f} FPS",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            # Window closed with the X button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:

        worker.stop()

        camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
