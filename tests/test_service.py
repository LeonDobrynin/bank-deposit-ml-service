import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.features import PrepareFeatures

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['1.0.0', '1.1.0'])
def release(request):
    return ROOT/'artifacts'/request.param


@pytest.fixture
def client(release):
    with TestClient(create_app(release)) as c:
        yield c


@pytest.fixture
def example(release):
    return json.loads((release/'etalon.json').read_text())[0]['features']


def test_health_and_info(client, release):
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['model_version'] == release.name
    assert health.json()['model_loaded'] is True
    info = client.get('/model-info')
    assert info.status_code == 200
    meta = info.json()
    assert meta['test_evaluated'] is False
    assert meta['split'] == {'train': 28823, 'validation': 6176, 'test': 6177}
    assert meta['model_sha256'] == hashlib.sha256((release/'model.joblib').read_bytes()).hexdigest()
    assert meta['data_sha256'] == hashlib.sha256((ROOT/'data/bank-additional-full.csv').read_bytes()).hexdigest()
    assert client.get('/docs').status_code == 200


def test_five_etalon_predictions(client, release):
    etalon = json.loads((release/'etalon.json').read_text())
    assert len(etalon) == 5
    model = joblib.load(release/'model.joblib')
    for row in etalon:
        response = client.post('/predict', json=row['features'])
        assert response.status_code == 200
        body = response.json()
        assert body['probability'] == pytest.approx(row['probability'], abs=1e-8)
        assert body['deposit'] == row['deposit']
        assert body['deposit'] == int(body['probability'] >= body['threshold'])
        direct = model.predict_proba(pd.DataFrame([row['features']]))[0, 1]
        assert direct == pytest.approx(body['probability'], abs=1e-12)


@pytest.mark.parametrize('change', [
    {'age': -1}, {'age': 101}, {'age': '34'}, {'age': True},
    {'job': 'typo'}, {'pdays': 1000}, {'previous': -1},
    {'duration': 100}, {'y': 'yes'}, {'loan': None}, {'poutcome': 'unknown'}])
def test_invalid_fields(client, example, change):
    assert client.post('/predict', json={**example, **change}).status_code == 422


def test_missing_and_malformed(client, example):
    example.pop('age')
    assert client.post('/predict', json=example).status_code == 422
    assert client.post('/predict', content='{', headers={'Content-Type': 'application/json'}).status_code == 422
    assert client.post('/predict', json=[]).status_code == 422


def test_unknown_category_is_valid(client, example):
    assert client.post('/predict', json={**example, 'job': 'unknown'}).status_code == 200


def test_unavailable_artifacts(tmp_path):
    example = json.loads((ROOT/'example.json').read_text())
    with TestClient(create_app(tmp_path)) as c:
        assert c.get('/health').status_code == 503
        assert c.get('/model-info').status_code == 503
        assert c.post('/predict', json=example).status_code == 503


def test_corrupted_artifact(tmp_path):
    (tmp_path/'model.joblib').write_bytes(b'broken')
    (tmp_path/'meta.json').write_text(json.dumps({'model_sha256': 'wrong'}))
    with TestClient(create_app(tmp_path)) as c:
        assert c.get('/health').status_code == 503


def test_preprocessing_does_not_mutate_input():
    source = pd.DataFrame([json.loads((ROOT/'example.json').read_text())])
    source['pdays'] = 999
    result = PrepareFeatures().fit_transform(source)
    assert result.pdays.iloc[0] == 0
    assert result.was_contacted_before.iloc[0] == 0
    assert source.pdays.iloc[0] == 999


def test_rollback_changes_policy_not_weights():
    old = ROOT/'artifacts/1.0.0'
    new = ROOT/'artifacts/1.1.0'
    assert (old/'model.joblib').read_bytes() == (new/'model.joblib').read_bytes()
    a = json.loads((old/'meta.json').read_text())
    b = json.loads((new/'meta.json').read_text())
    assert a['threshold'] == .5 and b['threshold'] == .1
    with TestClient(create_app(old)) as ca, TestClient(create_app(new)) as cb:
        sample = json.loads((ROOT/'example.json').read_text())
        # Force a boundary score to validate both threshold policies independently.
        class FixedModel:
            def predict_proba(self, X):
                return np.array([[.9, .1]])
        ca.app.state.model = cb.app.state.model = FixedModel()
        assert ca.post('/predict', json=sample).json()['deposit'] == 0
        assert cb.post('/predict', json=sample).json()['deposit'] == 1
