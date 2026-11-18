"""Deterministic preprocessing shared by training and inference."""
from sklearn.base import BaseEstimator, TransformerMixin

FEATURES = ['age', 'job', 'marital', 'education', 'default', 'housing',
            'loan', 'pdays', 'previous', 'poutcome']
NUMERIC = ['age', 'pdays', 'previous', 'was_contacted_before']
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


class PrepareFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X[FEATURES].copy()
        result['was_contacted_before'] = (result.pdays != 999).astype(int)
        result.loc[result.pdays == 999, 'pdays'] = 0
        return result
