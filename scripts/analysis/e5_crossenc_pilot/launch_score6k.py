"""Stage and launch ONE capped SageMaker Training job that scores the fixed-6k pairs with EXP-050 E5.

Writable prefixes for role aml2026-phase5-sagemaker-neural (inline policy NeuralSidecarArtifacts,
checked read-only before launch): phase5/runs/P5-SUB002-NGPU-* and
phase5/exp050-e5-large-v001/output/*. Inputs are staged under phase5/exp050-e5-large-v001/score6k-v001/
(readable by the role); outputs go to phase5/exp050-e5-large-v001/output/<job>/.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAIN = Path('/Users/earther/Desktop/Amazon ML Challange')
HERE = MAIN / 'outputs/analysis/e5_crossenc_pilot'
BUCKET = 'aml2026-ber-08be19ac500747'
ACCOUNT = '634393786318'
PROFILE = 'amamzon_01_a1_0'
REGION = 'us-east-1'
ROLE = 'aml2026-phase5-sagemaker-neural'
IMAGE = ('763104351884.dkr.ecr.us-east-1.amazonaws.com/'
         'huggingface-pytorch-training:2.9.0-transformers5.3.0-gpu-py312-cu130-ubuntu22.04')
RUN_ID = 'P5-E5-SCORE6K-001'
PREFIX = 'amazon-ml-2026/phase5/exp050-e5-large-v001/'
STAGE_PREFIX = PREFIX + 'score6k-v001/'
CKPT_KEY = PREFIX + 'output/P5-E5-18K-001/output/model.tar.gz'
INSTANCE = 'ml.g5.2xlarge'
PRICE_CAP = '2.0'          # USD/h ceiling (on-demand ml.g5.2xlarge training is below this)
MAX_RUNTIME = 2400         # seconds
COST_CAP = 4.0


def aws(*args: str) -> dict:
    env = dict(os.environ, AWS_SDK_UA_APP_ID='AWSSkill-SageMaker')
    cmd = ['aws', '--profile', PROFILE, '--region', REGION, '--no-cli-pager', *args, '--output', 'json']
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout) if out.stdout.strip() else {}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    code = HERE / 'stage/code'
    data = HERE / 'stage/data/pairs6k_e5.jsonl'
    driver = code / 'score_6k_driver.py'
    source = MAIN / 'scripts/analysis/e5_crossenc_pilot/score_6k_driver.py'
    if driver.exists() and sha(driver) != sha(source):
        raise FileExistsError('Staged driver differs from source; bump version')
    if not driver.exists():
        driver.write_bytes(source.read_bytes())
    config = {'run_id': RUN_ID, 'fold4': 'CLOSED', 'pairs_sha256': sha(data),
              'checkpoint_key': CKPT_KEY, 'instance': INSTANCE, 'max_runtime_seconds': MAX_RUNTIME,
              'batch_size': 512, 'max_length': 128, 'precision': 'fp16 autocast'}
    cfg = code / 'config.json'
    content = json.dumps(config, sort_keys=True, indent=2) + '\n'
    if cfg.exists() and cfg.read_text() != content:
        raise FileExistsError('Staged config differs; bump version')
    cfg.write_text(content)
    return config


def upload_immutable(path: Path, key: str) -> dict:
    digest = sha(path)
    enc = base64.b64encode(bytes.fromhex(digest)).decode()
    try:
        head = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key, '--checksum-mode', 'ENABLED')
    except subprocess.CalledProcessError:
        aws('s3api', 'put-object', '--bucket', BUCKET, '--key', key, '--body', str(path),
            '--if-none-match', '*', '--checksum-algorithm', 'SHA256', '--checksum-sha256', enc)
        head = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key, '--checksum-mode', 'ENABLED')
    if head.get('ChecksumSHA256') != enc or head['ContentLength'] != path.stat().st_size:
        raise ValueError(f'S3 checksum mismatch: {key}')
    return {'key': key, 'sha256': digest, 'bytes': path.stat().st_size}


def files() -> list[tuple[Path, str]]:
    return [(HERE / 'stage/code/score_6k_driver.py', STAGE_PREFIX + 'code/score_6k_driver.py'),
            (HERE / 'stage/code/config.json', STAGE_PREFIX + 'code/config.json'),
            (HERE / 'stage/data/pairs6k_e5.jsonl', STAGE_PREFIX + 'data/pairs6k_e5.jsonl')]


def channel(name: str, uri: str) -> dict:
    return {'ChannelName': name, 'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
            'S3Uri': uri, 'S3DataDistributionType': 'FullyReplicated'}}, 'InputMode': 'File'}


def request() -> dict:
    return {'TrainingJobName': RUN_ID,
            'AlgorithmSpecification': {'TrainingImage': IMAGE, 'TrainingInputMode': 'File',
                                       'ContainerEntrypoint': ['python3', '/opt/ml/input/data/code/score_6k_driver.py']},
            'RoleArn': f'arn:aws:iam::{ACCOUNT}:role/{ROLE}',
            'InputDataConfig': [channel('code', f's3://{BUCKET}/{STAGE_PREFIX}code/'),
                                channel('data', f's3://{BUCKET}/{STAGE_PREFIX}data/'),
                                channel('model', f's3://{BUCKET}/{CKPT_KEY}')],
            'OutputDataConfig': {'S3OutputPath': f's3://{BUCKET}/{PREFIX}output/'},
            'ResourceConfig': {'InstanceCount': 1, 'InstanceType': INSTANCE, 'VolumeSizeInGB': 30},
            'StoppingCondition': {'MaxRuntimeInSeconds': MAX_RUNTIME},
            'Tags': [{'Key': 'Project', 'Value': 'aml2026-phase5'}, {'Key': 'RunId', 'Value': RUN_ID},
                     {'Key': 'Experiment', 'Value': 'EXP-050-SCORE6K'}]}


def cost_guard() -> dict:
    sys.path.insert(0, str(MAIN / 'scripts/aws'))
    from cost_guard import guard
    before = Path.cwd()
    try:
        os.chdir(MAIN)
        return guard({'runtime_cap_minutes': MAX_RUNTIME / 60, 'total_cost_ceiling_usd': COST_CAP}, PRICE_CAP, '0.5')
    finally:
        os.chdir(before)


def launch() -> None:
    build()
    for path, key in files():
        head = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key, '--checksum-mode', 'ENABLED')
        if head.get('ChecksumSHA256') != base64.b64encode(bytes.fromhex(sha(path))).decode():
            raise ValueError(f'Staged input differs: {key}')
    aws('s3api', 'head-object', '--bucket', BUCKET, '--key', CKPT_KEY)
    role = aws('iam', 'get-role', '--role-name', ROLE)['Role']
    if role['Arn'] != f'arn:aws:iam::{ACCOUNT}:role/{ROLE}':
        raise ValueError('Unexpected role')
    jobs = aws('sagemaker', 'list-training-jobs', '--max-results', '100')['TrainingJobSummaries']
    if any(j['TrainingJobName'] == RUN_ID for j in jobs):
        raise RuntimeError('Job name already used; never launch a duplicate')
    budget = cost_guard()
    payload = request()
    print(json.dumps({'budget': budget, 'request': payload}, indent=2), flush=True)
    created = aws('sagemaker', 'create-training-job', '--cli-input-json', json.dumps(payload))
    (HERE / 'launch_receipt.json').write_text(json.dumps({'created': created, 'budget': budget,
                                                          'request': payload}, indent=2) + '\n')
    print(json.dumps({'created': created}), flush=True)


def main() -> None:
    os.environ['AWS_PROFILE'] = PROFILE
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    for a in ('plan', 'stage', 'launch', 'status'):
        g.add_argument('--' + a, action='store_true')
    args = p.parse_args()
    if args.plan:
        print(json.dumps({'config': build(), 'request': request()}, indent=2))
    elif args.stage:
        build()
        for path, key in files():
            print(json.dumps(upload_immutable(path, key)), flush=True)
    elif args.launch:
        launch()
    else:
        d = aws('sagemaker', 'describe-training-job', '--training-job-name', RUN_ID)
        keep = ('TrainingJobStatus', 'SecondaryStatus', 'FailureReason', 'TrainingTimeInSeconds',
                'BillableTimeInSeconds', 'ModelArtifacts')
        print(json.dumps({k: d.get(k) for k in keep}, indent=2, default=str))
        print(json.dumps([(t.get('Status'), t.get('StatusMessage')) for t in d.get('SecondaryStatusTransitions', [])],
                         indent=1, default=str))


if __name__ == '__main__':
    main()
