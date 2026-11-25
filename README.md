# Blind System

مشروع مساعد للكفيف (Blind System).

## الوحدات

| مجلد | الوصف |
|---|---|
| [`face-recognition/`](./face-recognition) | نظام التعرف على الوجوه: كاميرا + InsightFace + قاعدة بيانات الأشخاص. [README](./face-recognition/README.md) |
| [`emotion/`](./emotion) | **وحدة مستقلة**: كشف مشاعر الوجه (سعادة/حزن/غضب/...). [README](./emotion/README.md) |

قريبًا: وحدات (modules) إضافية أخرى ستُضاف في جذور الريبو.

## التشغيل

- سريع: انقر مزدوج على `face-recognition/run.cmd`
- VS Code: `Ctrl+Shift+B` → **Run face recognition**

## المتطلبات

- Python 3.11+
- `opencv-python`, `onnxruntime`, `insightface` — راجع `face-recognition/requirements.txt`
