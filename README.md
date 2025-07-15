
# 🧠 Breast Cancer Prediction API

تطبيق ويب بسيط مبني باستخدام **FastAPI** لتوقع إذا كانت أورام الثدي حميدة أو خبيثة باستخدام نموذج مدرب مسبقًا.

---

## 🚀 الميزات

- استخدام نموذج Decision Tree
- API مبني بـ FastAPI
- توثيق تلقائي عبر Swagger (OpenAPI)
- واجهة HTML لإدخال البيانات يدويًا
- قابل للتشغيل محليًا أو عبر Render

---

## 📁 هيكل المشروع

```
.
├── main.py
├── decision_tree_model.pkl
├── templates/
│   └── index.html
├── static/              # (اختياري، إن وجد ملفات CSS أو JS)
├── requirements.txt
└── README.md
```

---

## ⚙️ المتطلبات

- Python 3.8 أو أحدث
- pip

---

## 💻 خطوات التشغيل محليًا

1. **استنساخ المشروع:**

```bash
git clone https://github.com/YourUsername/cancer-predictor-api.git
cd cancer-predictor-api
```

2. **إنشاء بيئة افتراضية (اختياري):**

```bash
python -m venv venv
source venv/bin/activate      # على Linux/macOS
venv\Scripts\activate       # على Windows
```

3. **تثبيت المكتبات:**

```bash
pip install -r requirements.txt
```

4. **تشغيل السيرفر:**

```bash
uvicorn main:app --reload
```

5. **فتح التطبيق في المتصفح:**

- الصفحة الرئيسية: http://127.0.0.1:8000
- صفحة النموذج: http://127.0.0.1:8000/form
- توثيق API: http://127.0.0.1:8000/docs

---

## 🧠 النموذج

تم تدريب النموذج على بيانات سرطان الثدي من scikit-learn باستخدام خوارزمية Decision Tree، وتم حفظه باستخدام joblib في ملف `decision_tree_model.pkl`.

---

## 🌐 النشر على Render

1. اربط مستودع المشروع بـ GitHub.
2. افتح [https://render.com](https://render.com).
3. اختر **"Create New Web Service"**.
4. إعدادات Render:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`
5. اختر Python 3.9 أو أحدث.
6. اضغط "Deploy".

---

## ✅ المتطلبات في requirements.txt

```txt
fastapi
uvicorn
scikit-learn
joblib
jinja2
numpy
```

---

## 📬 تواصل

- GitHub: [@Omar-b381](https://github.com/Omar-b381)
- Email: omar.work381@gmail.com

---

## 📝 الرخصة

MIT © 2025
