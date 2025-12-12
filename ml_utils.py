from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# 1. كلاس تنظيف الحرارة
class TempCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # هذا السطر يقنع Scikit-Learn أن الكلاس تم تدريبه
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

# 2. كلاس استخراج الأمراض
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

# 3. كلاس التحويل (Mapping)
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
        # استخدام القاموس المخزن أو الافتراضي
        mapper = self.mapping_dict if self.mapping_dict else {}
        X[col_name] = X[col_name].astype(str).str.strip().map(mapper)
        return X