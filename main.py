from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os
import uvicorn

# استيراد الكلاسات من ml_utils (ضروري جداً لكي يفهم joblib الموديل)
from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper

app = FastAPI(title="NTI Covid Prediction API")

# ==========================================
# 1. إعدادات المسارات (Paths)
# ==========================================
# تحديد المجلد الحالي بدقة لضمان عمل الكود على Railway
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "automated_covid_model.pkl")
HTML_PATH = os.path.join(BASE_DIR, "index.html")

# ==========================================
# 2. تفعيل CORS (للسماح للمتصفح بالاتصال)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. تحميل الموديل
# ==========================================
model_pipeline = None
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully form: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==========================================
# 4. تعريف شكل البيانات (Pydantic Model)
# ==========================================
class PatientData(BaseModel):
    age: float
    temperature_C: str
    symptom_duration_days: float
    inflammatory_marker: float
    cough_level: str
    gender: str
    smoker_status: str
    clean_comorbidity_risk: str
    chronic_diseases: str
    city: str

    # إعدادات Pydantic V2
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 45,
            "temperature_C": "38.5C",
            "symptom_duration_days": 5,
            "inflammatory_marker": 12.5,
            "cough_level": "severe",
            "gender": "male",
            "smoker_status": "yes",
            "clean_comorbidity_risk": "high",
            "chronic_diseases": "asthma and diabetes",
            "city": "Cairo"
        }
    })

# ==========================================
# 5. الـ Endpoints (الرئيسي + التوقع)
# ==========================================

# الرابط الرئيسي: يعرض الداشبورد (index.html)
@app.get("/")
def read_root():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return {"error": "Dashboard file (index.html) not found. Please upload it."}

# رابط التوقع: يستقبل البيانات ويرجع النتيجة
@app.post("/predict")
def predict_covid(data: PatientData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")
    
    try:
        # تحويل البيانات لـ DataFrame
        # data.model_dump() هو البديل الحديث لـ .dict() في Pydantic V2
        # لكن .dict() ما زال يعمل، سنستخدم model_dump للأفضلية لو متاح، أو dict
        input_data = pd.DataFrame([data.dict()])
        
        # التوقع
        prediction = model_pipeline.predict(input_data)[0]
        result = "Positive" if prediction == 1 else "Negative"
        
        # محاولة جلب نسبة الثقة (Confidence)
        confidence = None
        try:
            probs = model_pipeline.predict_proba(input_data)[0]
            # لو النتيجة إيجابية نأخذ احتمال الـ 1، ولو سلبية نأخذ احتمال الـ 0
            confidence = float(probs[1]) if prediction == 1 else float(probs[0])
        except:
            pass

        return {
            "prediction": int(prediction), 
            "result_text": result,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==========================================
# 6. التشغيل المحلي (للتجربة)
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)