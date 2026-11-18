"""The same HTTP contract checks for local and container services."""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(url, version):
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url+path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    status, health = request('/health')
    assert status == 200 and health['model_version'] == version
    status, meta = request('/model-info')
    assert status == 200 and meta['model_version'] == version
    expected = json.loads((ROOT/'artifacts'/version/'etalon.json').read_text())
    responses = []
    for row in expected:
        status, body = request('/predict', row['features'])
        assert status == 200 and abs(body['probability']-row['probability']) < 1e-8
        assert body['deposit'] == row['deposit']
        assert body['model_version'] == version
        assert body['threshold'] == meta['threshold']
        responses.append(body)
    status, _ = request('/predict', {**expected[0]['features'], 'age': -1})
    assert status == 422
    return {'url': url, 'health': health, 'threshold': meta['threshold'],
            'five_predictions': responses, 'invalid_status': status, 'passed': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8077')
    p.add_argument('--version', default='1.1.0')
    args = p.parse_args()
    print(json.dumps(check(args.url, args.version), indent=2))
