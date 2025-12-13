from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import os
import uvicorn

from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper

app = FastAPI(title="NTI Covid Prediction API")

# ==========================================
# 1. إعدادات المسارات (Paths)
# ==========================================

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
    print(f"Model loaded successfully form: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

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

@app.get("/")
def read_root():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return {"error": "Dashboard file (index.html) not found. Please upload it."}

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
# 6. Analytics & Data Endpoints
# ==========================================

ANALYTICS_HTML_PATH = os.path.join(BASE_DIR, "charts.html")
CSV_PATH = os.path.join(BASE_DIR, "pcr_results_egyptian_applicants2020_dataset_final_version_dirty_extended_v3_with_city_and_markers.csv")

@app.get("/analytics")
def read_analytics():
    if os.path.exists(ANALYTICS_HTML_PATH):
        return FileResponse(ANALYTICS_HTML_PATH)
    return {"error": "Analytics file (charts.html) not found."}

@app.get("/api/stats")
def get_dataset_stats():
    try:
        # Load Data (using the same logic as training)
        df = pd.read_csv(CSV_PATH)
        
        # Helper to map categorical to numbers for aggregations if needed, but we focus on raw stats first
        # We need to map target for counts
        df['pcr_result_num'] = df['pcr_result'].map({'positive': 1, 'negative': 0})
        
        # 1. General counts
        total_samples = len(df)
        positive_cases = int(df['pcr_result_num'].sum())
        negative_cases = total_samples - positive_cases
        
        # 2. Averages per Class (Sick vs Healthy)
        # Group by Result
        # Convert numeric columns explicitly just in case
        numeric_cols = ['age', 'temperature_C', 'inflammatory_marker', 'symptom_duration_days']
        
        # Basic cleaning for stats (removing 'C' from temp if present effectively) - simple conversion
        # This mirrors clean logic but for quick stats visualization
        temp_df = df.copy()
        
        # Simple cleanup (assuming format is mostly clean or we handle errors)
        try:
           # Clean Temperature: Remove 'C', handle outliers roughly for visualization
           temp_df['temperature_C'] = temp_df['temperature_C'].astype(str).str.replace('C', '', case=False)
           temp_df['temperature_C'] = pd.to_numeric(temp_df['temperature_C'], errors='coerce')
        except:
            pass

        grouped = temp_df.groupby('pcr_result')[numeric_cols].mean().round(2)
        
        # Prepare Radar Data (Normalized relative to max for visual comparison, or raw?)
        # Let's send raw averages
        radar_data = {
            "positive": grouped.loc['positive'].to_dict() if 'positive' in grouped.index else {},
            "negative": grouped.loc['negative'].to_dict() if 'negative' in grouped.index else {}
        }
        
        # 3. City Distribution
        city_counts = df['city'].value_counts().head(5).to_dict()
        
        # 4. Scatter Data (Sample of 100 points to keep payload light)
        positive_sample = temp_df[temp_df['pcr_result'] == 'positive'].sample(min(50, positive_cases))
        negative_sample = temp_df[temp_df['pcr_result'] == 'negative'].sample(min(50, negative_cases))
        
        scatter_data = []
        for _, row in positive_sample.iterrows():
            if pd.notna(row['temperature_C']) and pd.notna(row['inflammatory_marker']):
                scatter_data.append({"x": row['temperature_C'], "y": row['inflammatory_marker'], "type": "positive"})
        
        for _, row in negative_sample.iterrows():
             if pd.notna(row['temperature_C']) and pd.notna(row['inflammatory_marker']):
                scatter_data.append({"x": row['temperature_C'], "y": row['inflammatory_marker'], "type": "negative"})

        # 5. Risk Factor breakdown
        risk_dist = temp_df.groupby(['clean_comorbidity_risk', 'pcr_result']).size().unstack(fill_value=0).to_dict(orient='index')

        return {
            "summary": {
                "total": total_samples,
                "positive": positive_cases,
                "negative": negative_cases
            },
            "averages": radar_data,
            "cities": city_counts,
            "scatter": scatter_data,
            "risk_distribution": risk_dist
        }

    except Exception as e:
        print(f"Stats Error: {e}")
        return {"error": str(e)}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)