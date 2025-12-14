import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from ml_utils import TempCleaner, DiseaseExtractor, ManualMapper, SmartAgeImputer

def main():
    print("Downloading data")
    file_id = "1PbnuTpG9utID_CLa1k88eTZw9tFuMeQq"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    df = pd.read_csv(url)

    df.drop_duplicates(inplace=True)
    df['pcr_result'] = df['pcr_result'].map({'negative': 0, 'positive': 1})
    X = df.drop(['pcr_result'], axis=1)
    y = df['pcr_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Building Pipeline")

    temp_pipeline = Pipeline([
        ('cleaner', TempCleaner()),
        ('imputer', KNNImputer(n_neighbors=5)), 
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

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
        ('age_fixer', SmartAgeImputer()), 
        ('preprocessor', preprocessor),
        ('model', SVC(probability=True)) 
    ])

    print("Training Model")
    full_pipeline.fit(X_train, y_train)

    train_acc = full_pipeline.score(X_train, y_train)
    print(f"\n🧠 Training Accuracy: {train_acc:.2%}")
    
    print("="*40)
    print("MODEL EVALUATION REPORT (TEST SET)")
    print("="*40)
    
    y_pred = full_pipeline.predict(X_test)

    # --- البداية: أضف هذا الجزء ---
    rec = recall_score(y_test, y_pred)
    print(f"⚠️ Recall Score (Sensitivity): {rec:.2%}")
    # --- النهاية ---

    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy:    {test_acc:.2%}")

    print("\n" + "="*40)
    print("MODEL EVALUATION REPORT")
    print("="*40)
    
    y_pred = full_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.2%}")
    print("-" * 30)

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives (Correct Healthy): {cm[0][0]}")
    print(f"False Positives (Wrongly Diagnosed Sick): {cm[0][1]}")
    print(f"False Negatives (Missed Cases - DANGEROUS): {cm[1][0]}")
    print(f"True Positives (Correct Sick): {cm[1][1]}")
    print("="*40 + "\n")

    joblib.dump(full_pipeline, 'automated_covid_model.pkl')
    print("Model Saved Successfully!")

if __name__ == "__main__":
    main()