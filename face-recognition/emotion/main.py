import time

import cv2

from camera import CameraStream

from emotion import EmotionDetector


WINDOW_NAME = "Emotion"


def main() -> None:
    print("=" * 50)
    print("Face Emotion Detection")
    print("=" * 50)

    detector = EmotionDetector()

    camera = CameraStream(mirror=True)

    if not camera.is_opened():
        print("Failed to open camera.")
        return

    camera.start()

    try:
        while True:
            frame = camera.read()

            if frame is None:
                time.sleep(0.005)
                continue

            emotion = detector.detect(frame)

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
