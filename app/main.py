import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import ClientFeatures, Prediction

ROOT = Path(__file__).resolve().parents[1]


def create_app(artifact_dir=None):
    directory = Path(artifact_dir or os.getenv('ARTIFACT_DIR', ROOT / 'artifacts' / '1.1.0'))

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        app.state.meta = None
        try:
            meta = json.loads((directory / 'meta.json').read_text())
            model_path = directory / 'model.joblib'
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta['model_sha256']:
                raise ValueError('Model checksum mismatch')
            model = joblib.load(model_path)
            app.state.model, app.state.meta = model, meta
        except Exception:
            logging.exception('Cannot load model artifacts')
        yield

    app = FastAPI(title='Прогноз отклика на банковский вклад', version='1.1.0', lifespan=lifespan)

    def ready():
        if app.state.model is None:
            raise HTTPException(status_code=503, detail='Model unavailable')

    @app.get('/health')
    def health():
        ready()
        return {'status': 'ok', 'model_loaded': True,
                'model_version': app.state.meta['model_version']}

    @app.get('/model-info')
    def model_info():
        ready()
        return app.state.meta

    @app.post('/predict', response_model=Prediction)
    def predict(client: ClientFeatures):
        ready()
        probability = float(app.state.model.predict_proba(pd.DataFrame([client.model_dump()]))[0, 1])
        threshold = app.state.meta['threshold']
        return Prediction(deposit=int(probability >= threshold), probability=probability,
                          threshold=threshold, model_version=app.state.meta['model_version'])

    return app


app = create_app()
