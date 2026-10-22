from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class DataFrameAgeScaler(BaseEstimator, TransformerMixin):
    # Standardize Age while preserving a mixed-type DataFrame for SMOTENC.
    def __init__(self, age_column="Age"):
        self.age_column = age_column

    def fit(self, X, y=None):
        self.mean_ = float(pd.to_numeric(X[self.age_column], errors="raise").mean())
        sd = float(pd.to_numeric(X[self.age_column], errors="raise").std(ddof=0))
        self.scale_ = sd if np.isfinite(sd) and sd > 0 else 1.0
        return self

    def transform(self, X):
        Xt = X.copy()
        Xt[self.age_column] = (pd.to_numeric(Xt[self.age_column], errors="raise") - self.mean_) / self.scale_
        return Xt
