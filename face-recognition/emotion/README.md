# Face Emotion Detection (وحدة مستقلة)

وحدة مستقلة لكشف مشاعر الوجه (`emotion`)، وهي منفصلة عن نظام التعرف على الوجه
(identity/face-recognition) على purpose — تقدر تبنى عليها وتطوّرها لوحدها.

## التشغيل

```powershell
cd face-recognition
python emotion\main.py
```

أو من VS Code: افتح `face-recognition/emotion/main.py` واضغط **F5**.

## الواجهة

- `EmotionDetector` هو اللي بيفلتر المنتج: بيتقبل `predictor` (callable) أو `model_path`.
- `emotion/main.py` بيعمل window اسمه `Emotion` ويطبع/يعرض الـ emotion للفريم الحالي.

## الحاجة لوحدة كاملة

- حمّل موديل المشاعر (مثل MediaPipe Emotion / ONNX Emotion) واحصل على `predictor`.
- بعد ما تكون محبك، تدمجش في الـ worker بتاع النظام الأساسي علشان تُرجع
  `(name, similarity, emotion)` جنبًا لجنب.

الخطوة القادمة المقترحة: تحميل موديل المشاعر وربطه بالـ `EmotionDetector`.
