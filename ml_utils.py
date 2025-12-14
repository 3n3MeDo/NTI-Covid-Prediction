from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TempCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        col_name = X.columns[0]
        X[col_name] = X[col_name].astype(str).str.replace('C', '', regex=False)
        X[col_name] = pd.to_numeric(X[col_name], errors='coerce')
        return X

class DiseaseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.diseases = ['asthma', 'diabetes', 'heart', 'hypertension', 'kidney']
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_out = pd.DataFrame(index=X.index)
        col_name = X.columns[0]
        col_data = X[col_name].astype(str).fillna('')
        for disease in self.diseases:
            X_out[disease] = col_data.apply(lambda x: 1 if disease in x else 0)
        return X_out

class ManualMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict=None):
        self.mapping_dict = mapping_dict
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        col_name = X.columns[0]
        mapper = self.mapping_dict if self.mapping_dict else {}
        X[col_name] = X[col_name].astype(str).str.strip().map(mapper)
        return X

class SmartAgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        df = X.copy()
        
        valid_age_df = df[df['age'] > 0].copy()
        
        valid_age_df['chronic_diseases'] = valid_age_df['chronic_diseases'].fillna('None')
        
        self.medians_1 = valid_age_df.groupby(['smoker_status', 'chronic_diseases'])['age'].median().to_dict()
        
        self.medians_2 = valid_age_df.groupby(['chronic_diseases'])['age'].median().to_dict()
        
        self.medians_3 = valid_age_df.groupby(['smoker_status'])['age'].median().to_dict()
        
        self.global_median = valid_age_df['age'].median()
        
        return self

    def transform(self, X):
        X = X.copy()
        
        def get_estimated_age(row):
            if row['age'] > 0:
                return row['age']
            
            smoker = row['smoker_status']
            disease = row['chronic_diseases'] if pd.notna(row['chronic_diseases']) else 'None'
            
            if (smoker, disease) in self.medians_1:
                return self.medians_1[(smoker, disease)]
            
            if disease in self.medians_2:
                return self.medians_2[disease]
            
            if smoker in self.medians_3:
                return self.medians_3[smoker]
            
            return self.global_median

        if 'age' in X.columns:
            X['age'] = X.apply(get_estimated_age, axis=1)
            
        return X