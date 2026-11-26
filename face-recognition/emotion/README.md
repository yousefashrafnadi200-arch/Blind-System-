# Face Emotion Detection (وحدة مستقلة)

وحدة مستقلة لكشف مشاعر الوجه، منفصلة عن نظام التعرف على الوجه (identity).

## الموديل

بتستخدم **FER+** بصيغة ONNX (8 مشاعر): `neutral, happiness, surprise, sadness,
anger, disgust, fear, contempt`.

نزّل الموديل مرة واحدة إلى:

```
face-recognition/emotion/models/emotion-ferplus-8.onnx
```

```powershell
$dir = "face-recognition/emotion/models"
New-Item -ItemType Directory -Force $dir | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" `
  -OutFile "$dir/emotion-ferplus-8.onnx"
```

> الموديل (~35 MB) مستثنى من git عبر `*.onnx` في `.gitignore`.

## التشغيل

```powershell
cd face-recognition
python -m emotion.main
```

أو من VS Code: افتح `face-recognition/emotion/main.py` → **F5** → config
`Python: face-recognition/emotion`.

يظهر window باسم `Emotion`، وفوقه آخر مشعر تم اكتشافه، وبيُطبع في الطرفية عند التغيّر:

```
Emotion: neutral
Emotion: no face
Emotion: happiness
```

**Esc** أو **q** للإغلاق.

## المكوّنات

| الملف | الوظيفة |
|---|---|
| `detector.py` | `EmotionDetector` + `OnnxEmotionBackend` (FER+ عبر onnxruntime) |
| `main.py` | تشغيل مستقل: كاميرا + عرض النتيجة |
| `models/` | مكان ملف الموديل (غير مُتتبَّع في git) |

## كيف يعمل

1. `InsightFace` (detection فقط) يحدد أكبر وجه في الفريم.
2. يُقصّ الوجه ويُحوَّل إلى رمادي 64×64.
3. `onnxruntime` يشغّل FER+ ويُرجع 8 درجات → softmax → أعلى مشعر.

## دمجها لاحقًا مع النظام الأساسي

```python
from emotion.detector import OnnxEmotionBackend
from emotion import EmotionDetector

emotion_detector = EmotionDetector(backend=OnnxEmotionBackend(MODEL_PATH))
emotion = emotion_detector.detect(frame)   # "happiness" / "no face" / ...
```

وبعدها يُرجَع `(name, similarity, emotion)` من الـ worker.
