import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# استيراد الكلاسات
from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper

def main():
    print("⏳ Downloading data...")
    file_id = "1PbnuTpG9utID_CLa1k88eTZw9tFuMeQq"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)

    # تجهيز الداتا
    df.drop_duplicates(inplace=True)
    df['pcr_result'] = df['pcr_result'].map({'negative': 0, 'positive': 1})
    X = df.drop(['pcr_result'], axis=1)
    y = df['pcr_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("⚙️ Building Pipeline...")

    temp_pipeline = Pipeline([
        ('cleaner', TempCleaner()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # دالة مساعدة
    def create_mapping_pipeline(mapping_dict):
        return Pipeline([
            ('mapper', ManualMapper(mapping_dict=mapping_dict)),
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

    cough_pipeline = create_mapping_pipeline({'mild': 1, 'moderate': 2, 'severe': 3})
    gender_pipeline = create_mapping_pipeline({'male': 0, 'female': 1})
    smoker_pipeline = create_mapping_pipeline({'no': 0, 'occasionally': 1, 'yes': 2})
    risk_pipeline = create_mapping_pipeline({'low': 1, 'medium': 2, 'high': 3})

    disease_pipeline = Pipeline([
        ('extractor', DiseaseExtractor())
    ])

    city_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # ============================================================
    # التعديل: إزالة n_jobs تماماً للاعتماد على الوضع الافتراضي الآمن
    # ============================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ('temp', temp_pipeline, ['temperature_C']),
            ('num', num_pipeline, ['age', 'symptom_duration_days', 'inflammatory_marker']),
            ('cough', cough_pipeline, ['cough_level']),
            ('gender', gender_pipeline, ['gender']),
            ('smoker', smoker_pipeline, ['smoker_status']),
            ('risk', risk_pipeline, ['clean_comorbidity_risk']),
            ('disease', disease_pipeline, ['chronic_diseases']),
            ('city', city_pipeline, ['city'])
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', SVC(probability=True))
    ])

    print("🚀 Training Model...")
    full_pipeline.fit(X_train, y_train)
    
    print("✅ Training Fit Complete. Calculating Score...")
    acc = full_pipeline.score(X_test, y_test)
    print(f"✅ Accuracy: {acc:.2f}")

    joblib.dump(full_pipeline, 'automated_covid_model.pkl')
    print("💾 Model Saved Successfully!")

if __name__ == "__main__":
    main()