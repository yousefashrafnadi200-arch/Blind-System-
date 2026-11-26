# Blind System

مشروع مساعد للكفيف (Blind System).

## الوحدات

كل وحدة في **مجلد مستقل** في جذر الريبو، وكل واحدة تشتغل لوحدها.

| مجلد | الوصف |
|---|---|
| [`face-recognition/`](./face-recognition) | نظام التعرف على الوجوه: كاميرا + InsightFace + قاعدة بيانات الأشخاص. [README](./face-recognition/README.md) |
| [`emotion/`](./emotion) | كشف مشاعر الوجه (فرحان/زعلان/غضبان/...). [README](./emotion/README.md) |

## التشغيل

| الوحدة | أسرع طريقة | VS Code |
|---|---|---|
| التعرف على الوجوه | `face-recognition/run.cmd` | `Ctrl+Shift+B` → **Run face recognition** أو `F5` → `Python: face-recognition` |
| المشاعر | `emotion/run.cmd` | `Ctrl+Shift+B` → **Run emotion** أو `F5` → `Python: emotion` |

> ⚠️ الكاميرا بتشتغل مع **نسخة واحدة** في نفس الوقت — اقفل وحدة قبل تشغيل التانية.

## المتطلبات

- Python 3.11+
- `face-recognition`: `numpy`, `opencv-python`, `onnxruntime`, `insightface` — [requirements](./face-recognition/requirements.txt)
- `emotion`: نفس الأساسيات + `pillow`, `arabic-reshaper`, `python-bidi` — [requirements](./emotion/requirements.txt)
