# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os
import uvicorn
# استيراد الكلاسات من ml_utils (ضروري جداً)
from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper

app = FastAPI(title="Covid Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "automated_covid_model.pkl")

model_pipeline = None

# محاولة تحميل الموديل
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

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

@app.post("/predict")
def predict_covid(data: PatientData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")
    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = model_pipeline.predict(input_data)[0]
        result = "Positive" if prediction == 1 else "Negative"
        
        # محاولة جلب النسبة
        confidence = None
        try:
            probs = model_pipeline.predict_proba(input_data)[0]
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
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)