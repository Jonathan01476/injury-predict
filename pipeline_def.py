"""Importable transformers shared by training and inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURES = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity', 'Recovery_Time']

class InjuryFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learn training medians and add body-size and load/recovery interactions."""
    def __init__(self, add_interactions=True):
        self.add_interactions = add_interactions

    def fit(self, X, y=None):
        frame = pd.DataFrame(X, columns=FEATURES).astype(float)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.medians_ = frame.median()
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        frame = pd.DataFrame(X, columns=FEATURES).astype(float).fillna(self.medians_).copy()
        if self.add_interactions:
            frame['Body_Size_Index'] = frame.Player_Weight / (frame.Player_Height / 100) ** 2
            frame['Load_Recovery_Ratio'] = frame.Training_Intensity / frame.Recovery_Time
            frame['Prior_Injury_Load'] = frame.Previous_Injuries * frame.Training_Intensity
        return frame
