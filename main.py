from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os
import uvicorn
import numpy as np

# استدعاء الأدوات الخاصة بنا
from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper, SmartAgeImputer

app = FastAPI(title="NTI Covid Prediction API")

# ==========================================
# 1. إعدادات المسارات (Paths)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "automated_covid_model.pkl")
HTML_PATH = os.path.join(BASE_DIR, "index.html")
ANALYTICS_HTML_PATH = os.path.join(BASE_DIR, "charts.html")
CSV_PATH = os.path.join(BASE_DIR, "pcr_results_egyptian_applicants2020_dataset_final_version_dirty_extended_v3_with_city_and_markers.csv")

# ==========================================
# 2. تفعيل CORS
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
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==========================================
# 4. تعريف البيانات
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
# 5. Endpoints
# ==========================================

@app.get("/")
def read_root():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return {"error": "Dashboard file (index.html) not found."}

@app.get("/analytics")
def read_analytics():
    if os.path.exists(ANALYTICS_HTML_PATH):
        return FileResponse(ANALYTICS_HTML_PATH)
    return {"error": "Analytics file (charts.html) not found."}

@app.get("/{filename}")
async def read_static_file(filename: str):
    """
    Serve static files (images, css, js) if they exist in the root directory.
    Restricted to specific extensions for security.
    """
    file_path = os.path.join(BASE_DIR, filename)
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".css", ".js"}
    
    # Check extension
    _, ext = os.path.splitext(filename)
    if ext.lower() not in allowed_extensions:
        # Fallback: maybe let FastAPI handle 404 naturally or return specific error
        # For now, we only serve known safe types from root
        raise HTTPException(status_code=404, detail="File type not allowed or file not found")

    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/predict")
def predict_covid(data: PatientData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")
    
    try:
        input_data = pd.DataFrame([data.model_dump()])
        
        probs = model_pipeline.predict_proba(input_data)[0]
        
        # Logic: Select class with highest probability
        if probs[1] > probs[0]:
            prediction = 1
            confidence = float(probs[1])
        else:
            prediction = 0
            confidence = float(probs[0])
            
        result_text = "Positive" if prediction == 1 else "Negative"
        
        return {
            "prediction": int(prediction), 
            "result_text": result_text,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/stats")
def get_dataset_stats():
    """
    يقوم هذا الـ Endpoint بحساب جميع الإحصائيات اللازمة للرسومات البيانية
    بما في ذلك التوزيعات الجديدة (Age, Gender, Smoker, Disease Count).
    """
    try:
        if not os.path.exists(CSV_PATH):
            return {"error": "Dataset CSV file not found."}

        # Load Data
        df = pd.read_csv(CSV_PATH)
        
        # --- Preprocessing for Stats ---
        # Map Result to numbers for counting
        df['pcr_result_num'] = df['pcr_result'].map({'positive': 1, 'negative': 0})
        
        # Clean Temp (Simple cleanup for visualization)
        df['temperature_C'] = df['temperature_C'].astype(str).str.replace('C', '', regex=False)
        df['temperature_C'] = pd.to_numeric(df['temperature_C'], errors='coerce')

        # منطق حساب عدد الأمراض المزمنة
        def count_diseases(val):
            if pd.isna(val) or str(val).lower() in ['none', 'nan', '']:
                return 0
            return len(str(val).split(','))
        
        df['disease_count'] = df['chronic_diseases'].apply(count_diseases)

        # تقسيم الأعمار إلى فئات (Age Groups)
        bins = [0, 18, 30, 45, 60, 80, 120]
        labels = ['0-18', '19-30', '31-45', '46-60', '61-80', '80+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # --- Aggregations (تجميع البيانات) ---

        # 1. Summary KPIs
        total_samples = len(df)
        positive_cases = int(df['pcr_result_num'].sum())
        negative_cases = total_samples - positive_cases

        # 2. Gender Distribution (Grouped by Result) -> For Bar Chart
        gender_dist = df.groupby(['gender', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        # 3. Smoker Distribution -> For Bar Chart
        smoker_dist = df.groupby(['smoker_status', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        # 4. Age Group Distribution -> For Bar Chart
        age_dist = df.groupby(['age_group', 'pcr_result'], observed=False).size().unstack(fill_value=0).to_dict(orient='index')

        # 5. Disease Count Distribution -> For Bar Chart
        disease_count_dist = df.groupby(['disease_count', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        # 6. City Distribution (Detailed Positive/Negative) -> For Bar Chart
        city_dist = df.groupby(['city', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        # 7. Radar Data (Averages)
        numeric_cols = ['age', 'temperature_C', 'inflammatory_marker', 'symptom_duration_days']
        grouped_avg = df.groupby('pcr_result')[numeric_cols].mean().round(2)
        radar_data = {
            "positive": grouped_avg.loc['positive'].to_dict() if 'positive' in grouped_avg.index else {},
            "negative": grouped_avg.loc['negative'].to_dict() if 'negative' in grouped_avg.index else {}
        }
        
        # 8. Scatter Sample (Temperature vs Marker)
        positive_sample = df[df['pcr_result'] == 'positive'].sample(min(50, positive_cases))
        negative_sample = df[df['pcr_result'] == 'negative'].sample(min(50, negative_cases))
        scatter_data = []
        for _, row in positive_sample.iterrows():
            if pd.notna(row['temperature_C']) and pd.notna(row['inflammatory_marker']):
                scatter_data.append({"x": row['temperature_C'], "y": row['inflammatory_marker'], "type": "positive"})
        for _, row in negative_sample.iterrows():
             if pd.notna(row['temperature_C']) and pd.notna(row['inflammatory_marker']):
                scatter_data.append({"x": row['temperature_C'], "y": row['inflammatory_marker'], "type": "negative"})

        # 9. Risk Polar (Existing)
        risk_dist = df.groupby(['clean_comorbidity_risk', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        return {
            "summary": { "total": total_samples, "positive": positive_cases, "negative": negative_cases },
            "distributions": {
                "gender": gender_dist,
                "smoker": smoker_dist,
                "age": age_dist,
                "disease_count": disease_count_dist,
                "city": city_dist
            },
            "averages": radar_data,
            "scatter": scatter_data,
            "risk_distribution": risk_dist
        }

    except Exception as e:
        print(f"Stats Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)