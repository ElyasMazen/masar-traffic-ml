# مسار — Masar Simple Prototype

نسخة مبسطة من مشروع **مسار** مخصصة للعرض كبروتوتايب فقط.

الواجهة الحالية تركز على الأشياء الأساسية فقط:

- اختيار المدينة: **Jeddah** (بيانات مسار من TomTom) أو **New York City** (بيانات مرجعية).
- البحث عن طريق أو اختيار الطريق من الخريطة.
- اختيار التاريخ والساعة.
- خريطة تفاعلية تتحرك إلى الطريق المختار.
- نتيجة بسيطة تشمل:
  - Expected Traffic
  - Traffic Level
  - Accident Risk
  - Road Safety Score
  - Similar Past Crashes
  - 12-hour traffic simulation

تم حذف الأقسام الزائدة من الواجهة مثل Police Simulation وAccident Intelligence وGovernment Center وAI Analyst حتى تبقى النسخة واضحة وخفيفة كبروتوتايب.

## التقنيات
- Frontend: Next.js + TypeScript
- Backend: FastAPI + Python
- Map: MapLibre GL JS + OpenStreetMap
- Machine Learning: scikit-learn
- Data: Historical NYC traffic + crash data

## التشغيل لأول مرة

الواجهة موجودة داخل مستودع `masar-traffic-ml` في المجلد `web/`. كل الأوامر التالية من داخل `web/`
(أو شغّل `SETUP_WINDOWS.bat` ثم `RUN_PROJECT.bat`).

### Backend
```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\train_model.py
python -m uvicorn app.main:app --reload
```

### Frontend
افتح Terminal ثاني:
```powershell
cd frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm install
npm run dev
```

ثم افتح:
- http://localhost:3000

## التشغيل في المرات القادمة

### Terminal 1 — Backend
```powershell
cd backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

### Terminal 2 — Frontend
```powershell
cd frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm run dev
```

## Jeddah (بيانات مسار)

اختر **Jeddah** من قائمة City. تستخدم هذه الواجهة بيانات TomTom التي جمعناها لجدة (3 طرق، 10 نقاط) ونموذجين:

- **Time-of-day** — يتوقع السرعة ومستوى الازدحام لأي نقطة/تاريخ/ساعة (البطاقات، ألوان الخريطة، رسم الـ12 ساعة). يُختار تلقائياً الأفضل بين نموذج الوقت والمتوسط التاريخي.
- **Random Forest قصير المدى** — مستوى الازدحام بعد آخر قراءة، عند الأفق الأكثر فائدة (حالياً +60 دقيقة)، ويُعرض دائماً بجانب الـpersistence baseline.

كل التدريب والتقييم في جذر المستودع (`train_masar.py`). الـbackend لا ينسخ شيئاً: يستورد `masar_features.py` من الجذر،
ويقرأ `models/` و`outputs/results.json` والـsnapshot الذي تدرّب عليه (`data/snapshots/`، خارج git) مباشرة.

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\build_jeddah.py              # يتحقق من الملفات، ويدرّب فقط إذا كان شيء ناقصاً
python scripts\build_jeddah.py --retrain    # تدريب جديد من ملف GitHub الخام
```

التدريب يستخدم بيئة الجذر (`..\.venv`) إن وُجدت. لإنشائها مرة واحدة من جذر المستودع:
`py -m venv .venv` ثم `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. أعد تشغيل الـbackend بعد التدريب.
لا يُستدعى TomTom API أبداً، ولا يُكتب شيء في مستودع البيانات.

## ملاحظة
هذه النسخة **Prototype** وليست نظام مرور مباشر. النتائج تعتمد على بيانات تاريخية من نيويورك وليست Live Traffic.
