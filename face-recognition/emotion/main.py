import os
import sys
import time

import cv2

# Make the parent folder (face-recognition/) importable when this file is run
# directly as a script, e.g. `python emotion/main.py` or VS Code's F5.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from camera import CameraStream

from emotion import EmotionDetector
from emotion.arabic_text import draw_text
from emotion.detector import OnnxEmotionBackend, to_arabic


WINDOW_NAME = "Emotion"

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "enet_b0_8_best_vgaf.onnx",
)


def _use_utf8_console() -> None:
    """Keep Arabic prints from crashing on the Windows console."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _show_label(frame, label_ar: str, label_en: str) -> None:
    try:
        draw_text(frame, label_ar, font_size=34, color=(0, 255, 0))
    except Exception:
        # No Arabic-capable font / Pillow missing: fall back to the Latin label.
        height = frame.shape[0]
        cv2.putText(
            frame,
            f"Emotion: {label_en}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


def main() -> None:
    _use_utf8_console()

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

            label_ar = to_arabic(emotion)

            if emotion != last_emotion:
                print(f"Emotion: {label_ar} ({emotion})")
                last_emotion = emotion

                try:
                    cv2.setWindowTitle(WINDOW_NAME, f"Emotion - {label_ar}")
                except Exception:
                    pass

            _show_label(frame, label_ar, emotion)

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
