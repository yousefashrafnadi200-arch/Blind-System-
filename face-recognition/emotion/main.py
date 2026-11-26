import os
import time

import cv2

from camera import CameraStream

from emotion import EmotionDetector
from emotion.detector import OnnxEmotionBackend


WINDOW_NAME = "Emotion"

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "emotion-ferplus-8.onnx",
)


def main() -> None:
    print("=" * 50)
    print("Face Emotion Detection")
    print("=" * 50)

    print(f"\nLoading emotion model: {MODEL_PATH}")

    detector = EmotionDetector(
        backend=OnnxEmotionBackend(MODEL_PATH)
    )

    print("Emotion model loaded.")

    camera = CameraStream(mirror=True)

    if not camera.is_opened():
        print("Failed to open camera.")
        return

    camera.start()

    last_emotion = None

    try:
        while True:
            frame = camera.read()

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                emotion = detector.detect(frame)
            except Exception as exc:
                emotion = "unavailable"

                if emotion != last_emotion:
                    print(f"Emotion detection unavailable: {exc}")

            if emotion != last_emotion:
                print(f"Emotion: {emotion}")
                last_emotion = emotion

            height, width = frame.shape[:2]
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
