# Face Emotion Detection (وحدة مستقلة)

وحدة مستقلة لكشف مشاعر الوجه، منفصلة عن نظام التعرف على الوجه (identity).

## الموديل

**HSEmotion (EfficientNet-B0)** مُدرَّب على AffectNet، بصيغة ONNX.

8 مشاعر (بترتيب المخرجات):

```
Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
```

نزّل الموديل مرة واحدة إلى:

```
face-recognition/emotion/models/enet_b0_8_best_vgaf.onnx
```

```powershell
$dir = "face-recognition/emotion/models"
New-Item -ItemType Directory -Force $dir | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx" `
  -OutFile "$dir/enet_b0_8_best_vgaf.onnx"
```

> الموديل (~16 MB) مستثنى من git عبر `*.onnx` في `.gitignore`.

## التشغيل

```powershell
cd face-recognition
python -m emotion.main
```

أو مباشرة:

```powershell
python emotion\main.py
```

أو من VS Code: افتح `face-recognition/emotion/main.py` → **F5** → config
`Python: face-recognition/emotion`.

نافذة `Emotion` بتعرض آخر مشعر، والطرفية بتطبع عند كل تغيّر:

```
Emotion: Happiness
Emotion: Neutral
Emotion: Surprise
```

**Esc** أو **q** للإغلاق.

## المكوّنات

| الملف | الوظيفة |
|---|---|
| `detector.py` | `EmotionDetector` + `OnnxEmotionBackend` (HSEmotion عبر onnxruntime) |
| `main.py` | تشغيل مستقل: كاميرا + عرض النتيجة |
| `models/` | ملف الموديل (غير مُتتبَّع في git) |

## كيف يعمل

1. **InsightFace** (detection فقط) يحدد أكبر وجه + 5 نقاط محاذاة.
2. يُحاذى الوجه (`norm_crop`) ويُصغَّر إلى 224×224 RGB.
3. تطبيع ImageNet: `(x/255 - mean) / std`.
4. **onnxruntime** يشغّل الموديل → 8 درجات → softmax → أعلى مشعر.

## ملاحظات

- الموديل بيحتاج وجه **محاذى**؛ القصّ المباشر من الـ bbox (بدون محاذاة) بيقلل الدقة
  بشكل كبير — وده اللي كان بيخلي النتيجة `Neutral` دايمًا في نسخة FER+ القديمة.
- `mediapipe` مش مستخدم هنا (إصداره الحالي بدون emotion recognizer).

## دمجها لاحقًا مع النظام الأساسي

```python
from emotion.detector import OnnxEmotionBackend
from emotion import EmotionDetector

emotion_detector = EmotionDetector(backend=OnnxEmotionBackend(MODEL_PATH))
emotion = emotion_detector.detect(frame)   # "Happiness" / "no face" / ...
```

وبعدها يُرجَع `(name, similarity, emotion)` من الـ worker.
