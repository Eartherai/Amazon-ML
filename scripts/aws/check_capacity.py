"""Read-only AWS inventory. Stores sanitized responses; never reads credentials."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import subprocess


def call(profile, args):
    command = ['aws', '--profile', profile, '--region', 'us-east-1', '--no-cli-pager', '--cli-connect-timeout', '8', '--cli-read-timeout', '15', *args, '--output', 'json']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=55, env={**os.environ, 'AWS_MAX_ATTEMPTS':'1'})
        value = (json.loads(result.stdout) if result.stdout.strip() else {'ok': True}) if result.returncode == 0 else {'error': result.stderr.strip()}
        return json.loads(re.sub(r'\b\d{12}\b', 'REDACTED_ACCOUNT', json.dumps(value)))
    except Exception as exc:
        return {'error': str(exc)}


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    out = Path('artifacts/cloud/phase5') / now.strftime('capacity-%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    profiles = subprocess.check_output(['aws','configure','list-profiles'], text=True).splitlines()
    jobs = {}
    for profile in profiles:
        jobs[(profile,'identity')] = ['sts','get-caller-identity']
        jobs[(profile,'aliases')] = ['iam','list-account-aliases']
        jobs[(profile,'ec2_quotas')] = ['service-quotas','list-service-quotas','--service-code','ec2']
        jobs[(profile,'sagemaker_quotas')] = ['service-quotas','list-service-quotas','--service-code','sagemaker']
        jobs[(profile,'instances')] = ['ec2','describe-instances','--query','Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name}']
        jobs[(profile,'offerings')] = ['ec2','describe-instance-type-offerings','--filters','Name=instance-type,Values=r8i.*,r7i.*,c7i.*,g6.*,g6e.*,p4d.*,p5.*,p5en.*']
        jobs[(profile,'roles')] = ['iam','list-instance-profiles','--query','InstanceProfiles[].InstanceProfileName']
        jobs[(profile,'s3')] = ['s3api','head-bucket','--bucket','aml2026-ber-08be19ac500747']
    report = {'timestamp':now.isoformat(),'region':'us-east-1','profiles':{},'credits':'Unverified; reported approximately $200 is not an API balance.'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(call,p,args):(p,k) for (p,k),args in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            p,k=futures[future]; value=future.result()
            report['profiles'].setdefault(p,{})[k]=value
            print(p,k, 'ERROR' if isinstance(value,dict) and 'error' in value else 'OK',flush=True)
    (out/'inventory.json').write_text(json.dumps(report,indent=2))
    print(out)

if __name__ == '__main__':
    main()
