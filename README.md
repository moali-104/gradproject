# ASD Prediction API 🧠

Flask API للتنبؤ بـ Autism Spectrum Disorder باستخدام 3 موديلز.

## الإعداد والتشغيل

### 1. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 2. تشغيل السيرفر
```bash
python app.py
```
السيرفر هيشتغل على: `http://localhost:5000`

---

## الـ Endpoints

### `GET /`
عرض معلومات الـ API والـ features المطلوبة.

### `POST /predict/adaboost`
التنبؤ باستخدام AdaBoost.

### `POST /predict/random_forest`
التنبؤ باستخدام Random Forest.

### `POST /predict/all`
التنبؤ بالـ 3 موديلز مع Majority Vote.

---

## مثال على الـ Request

```json
{
  "A1": 1, "A2": 0, "A3": 1, "A4": 1, "A5": 0,
  "A6": 1, "A7": 0, "A8": 1, "A9": 1, "A10": 1,
  "Age": 25,
  "Sex": 1,
  "Jauundice": 0,
  "Family_ASD": 1
}
```

### باستخدام curl:
```bash
curl -X POST http://localhost:5000/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "A1":1,"A2":0,"A3":1,"A4":1,"A5":0,
    "A6":1,"A7":0,"A8":1,"A9":1,"A10":1,
    "Age":25,"Sex":1,"Jauundice":0,"Family_ASD":1
  }'
```

### الـ Response:
```json
{
  "results": {
    "adaboost": {
      "prediction": 1,
      "label": "ASD Positive",
      "probability": { "negative": 0.32, "positive": 0.68 }
    },
    "random_forest": {
      "prediction": 1,
      "label": "ASD Positive",
      "probability": { "negative": 0.15, "positive": 0.85 }
    }
  },
  "majority_vote": {
    "prediction": 1,
    "label": "ASD Positive"
  }
}
```

---

## الـ Features

| Feature | الوصف |
|---------|-------|
| A1 - A10 | أسئلة الاستبيان (0 أو 1) |
| Age | عمر المريض |
| Sex | الجنس (0=Female, 1=Male) |
| Jauundice | هل كان عنده يرقان وقت الولادة (0/1) |
| Family_ASD | هل في تاريخ عائلي لـ ASD (0/1) |

---

## ملاحظة
موديل Gradient Boosting معلق مؤقتاً لأنه محتاج `scikit-learn==1.3.0` بالظبط.
بعد تثبيت requirements.txt هيشتغل عادي.
