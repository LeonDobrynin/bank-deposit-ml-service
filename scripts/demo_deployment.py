"""Verify local HTTP, deploy v1, upgrade to v1.1, roll back to v1.

Run after both images are built: python scripts/demo_deployment.py
Only creates/removes its own uniquely named containers; binds loopback only.
"""
import json
import subprocess
import sys
import time
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from check_http import check

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'


def wait_ready(url):
    for _ in range(60):
        try:
            with urllib.request.urlopen(url+'/health', timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(.5)
    raise RuntimeError('Service did not become ready: '+url)


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def main():
    EVIDENCE.mkdir(exist_ok=True)
    log = {'date': datetime.now(timezone.utc).isoformat(), 'stages': []}
    with (EVIDENCE/'local-server.log').open('w') as stream:
        local = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                                  '--host', '127.0.0.1', '--port', '18077'],
                                 cwd=ROOT, stdout=stream, stderr=stream)
        try:
            wait_ready('http://127.0.0.1:18077')
            log['local'] = check('http://127.0.0.1:18077', '1.1.0')
        finally:
            local.terminate()
            local.wait(timeout=15)
    name = 'bank-deposit-demo-'+uuid.uuid4().hex[:10]
    for stage, version in [('initial', '1.0.0'), ('upgrade', '1.1.0'), ('rollback', '1.0.0')]:
        image = 'bank-deposit-group6:'+version
        cid = docker('run', '-d', '--name', name, '-p', '127.0.0.1:18078:8077', image)
        try:
            wait_ready('http://127.0.0.1:18078')
            result = check('http://127.0.0.1:18078', version)
            for _ in range(30):
                state = json.loads(docker('inspect', name))[0]
                if state['State']['Health']['Status'] == 'healthy':
                    break
                time.sleep(.5)
            assert state['State']['Health']['Status'] == 'healthy'
            result.update(stage=stage, container_id=cid, image_id=state['Image'],
                          docker_health=state['State']['Health']['Status'])
            log['stages'].append(result)
            (EVIDENCE/f'container-{stage}.log').write_text(docker('logs', name))
        finally:
            docker('rm', '-f', name)
    old, new, rollback = log['stages']
    assert old['image_id'] == rollback['image_id'] != new['image_id']
    assert old['five_predictions'] == rollback['five_predictions']
    for a, b in zip(log['local']['five_predictions'], new['five_predictions']):
        assert a['deposit'] == b['deposit']
        assert abs(a['probability']-b['probability']) < 1e-8
    log['local_container_parity'] = True
    log['rollback_verified'] = True
    (EVIDENCE/'deployment.json').write_text(json.dumps(log, indent=2)+'\n')
    print('PASS: local HTTP, both containers, invalid requests, parity and rollback')


if __name__ == '__main__':
    main()
