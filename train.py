"""Train both release artifacts: python train.py. Test set remains untouched."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.features import PrepareFeatures, FEATURES, NUMERIC, CATEGORICAL

ROOT = Path(__file__).resolve().parent


def train():
    source = ROOT / 'data' / 'bank-additional-full.csv'
    raw = pd.read_csv(source, sep=';')
    clean = raw.drop_duplicates(keep='first')
    a, b = int(len(clean)*.7), int(len(clean)*.85)
    training, validation = clean.iloc[:a], clean.iloc[a:b]
    y = validation.y.eq('yes').astype(int).to_numpy()
    model = Pipeline([
        ('prepare', PrepareFeatures()),
        ('preprocessing', ColumnTransformer([
            ('numeric', StandardScaler(), NUMERIC),
            ('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)])),
        ('model', LogisticRegression(C=1, max_iter=2000, solver='lbfgs', random_state=42))])
    model.fit(training[FEATURES], training.y.eq('yes').astype(int))
    scores = model.predict_proba(validation[FEATURES])[:, 1]
    k = int(len(validation)*.1)
    tie = np.random.default_rng(42).random(len(validation))
    selected = np.zeros(len(validation), dtype=bool)
    selected[np.lexsort((tie, -scores))[:k]] = True
    common = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'target': 'y=yes (term deposit subscription)', 'features': FEATURES,
        'data_version': 'uci-bank-additional-full',
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'duplicates_removed': len(raw)-len(clean),
        'split': {'train': a, 'validation': b-a, 'test': len(clean)-b},
        'test_evaluated': False,
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'metrics': {'validation_precision_at_10pct': float(precision_score(y, selected)),
                    'validation_recall_at_10pct': float(recall_score(y, selected)),
                    'validation_average_precision': float(average_precision_score(y, scores)),
                    'selected_top10': k, 'true_positives_top10': int(y[selected].sum())}}
    # Two release policies, same trained weights. Rollback restores threshold 0.5.
    # Release 1.1.0 uses the threshold 0.1 already studied in practice 3.
    for version, threshold in [('1.0.0', .5), ('1.1.0', .1)]:
        out = ROOT / 'artifacts' / version
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / 'model.joblib')
        mask = scores >= threshold
        meta = {**common, 'model_version': version, 'threshold': threshold,
                'release_change': 'decision threshold only; identical trained weights',
                'model_sha256': hashlib.sha256((out/'model.joblib').read_bytes()).hexdigest(),
                'threshold_metrics': {'selected': int(mask.sum()),
                    'precision': float(precision_score(y, mask, zero_division=0)),
                    'recall': float(recall_score(y, mask, zero_division=0))}}
        # Five records spanning the score range, including both decision classes.
        order = np.argsort(scores)
        positions = [order[0], order[len(order)//2], order[-3], order[-2], order[-1]]
        etalon = [{'features': validation.iloc[int(i)][FEATURES].to_dict(),
                   'probability': float(scores[i]), 'deposit': int(scores[i] >= threshold)} for i in positions]
        (out/'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
        (out/'etalon.json').write_text(json.dumps(etalon, ensure_ascii=False, indent=2)+'\n')
    sample = validation.iloc[int(order[-1])][FEATURES].to_dict()
    (ROOT/'example.json').write_text(json.dumps(sample, indent=2)+'\n')
    print(json.dumps(common, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    train()
