"""Private versioned S3 foundation via AWS CLI, without reading credentials.

Conditional writes and verified SHA256 prevent accidental overwrite. The raw
prefix also denies deletes and unconditional creates at the bucket policy layer.
An account administrator can still change that policy; this is not Object Lock.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

STATE=Path('artifacts/cloud/phase3/s3_state.json')
REGION='us-east-1'
PREFIX='amazon-ml-2026/'


def aws(*args:str, allow_missing=False)->dict:
    r=subprocess.run(['aws',*args,'--region',REGION,'--output','json','--no-cli-pager'],capture_output=True,text=True)
    if r.returncode:
        if allow_missing and ('(404)' in r.stderr or 'Not Found' in r.stderr or 'NoSuchKey' in r.stderr):return {}
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def setup()->dict:
    ident=aws('sts','get-caller-identity')
    if STATE.exists():
        state=json.loads(STATE.read_text())
        if state['account_suffix']!=ident['Account'][-4:]:raise RuntimeError('Unexpected active account')
    else:
        state={'bucket':'aml2026-ber-'+uuid.uuid4().hex[:14],'region':REGION,'prefix':PREFIX,'account_suffix':ident['Account'][-4:],'created_at':datetime.now(timezone.utc).isoformat()}
        aws('s3api','create-bucket','--bucket',state['bucket'],'--object-ownership','BucketOwnerEnforced')
        STATE.parent.mkdir(parents=True,exist_ok=True);STATE.write_text(json.dumps(state,indent=2))
    bucket=state['bucket'];root='arn:aws:s3:::'+bucket
    aws('s3api','put-public-access-block','--bucket',bucket,'--public-access-block-configuration',json.dumps(dict.fromkeys(['BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets'],True)))
    aws('s3api','put-bucket-encryption','--bucket',bucket,'--server-side-encryption-configuration',json.dumps({'Rules':[{'ApplyServerSideEncryptionByDefault':{'SSEAlgorithm':'AES256'}}]}))
    aws('s3api','put-bucket-versioning','--bucket',bucket,'--versioning-configuration','Status=Enabled')
    policy={'Version':'2012-10-17','Statement':[
       {'Sid':'DenyInsecureTransport','Effect':'Deny','Principal':'*','Action':'s3:*','Resource':[root,root+'/*'],'Condition':{'Bool':{'aws:SecureTransport':'false'}}},
       {'Sid':'RequireRawConditionalCreation','Effect':'Deny','Principal':'*','Action':'s3:PutObject','Resource':root+'/'+PREFIX+'raw/*','Condition':{'Null':{'s3:if-none-match':'true'},'Bool':{'s3:ObjectCreationOperation':'true'}}},
       {'Sid':'ProtectRawDeletes','Effect':'Deny','Principal':'*','Action':['s3:DeleteObject','s3:DeleteObjectVersion'],'Resource':root+'/'+PREFIX+'raw/*'}]}
    aws('s3api','put-bucket-policy','--bucket',bucket,'--policy',json.dumps(policy))
    aws('s3api','put-bucket-lifecycle-configuration','--bucket',bucket,'--lifecycle-configuration',json.dumps({'Rules':[{'ID':'AbortIncompleteUploads','Status':'Enabled','Filter':{'Prefix':PREFIX},'AbortIncompleteMultipartUpload':{'DaysAfterInitiation':1}}]}))
    aws('s3api','put-bucket-tagging','--bucket',bucket,'--tagging',json.dumps({'TagSet':[{'Key':'Project','Value':'AmazonML2026'},{'Key':'Purpose','Value':'PrivateCompetitionData'}]}))
    (STATE.parent/'bucket_policy.json').write_text(json.dumps(policy,indent=2))
    checks={k:aws('s3api',cmd,'--bucket',bucket) for k,cmd in [('public_access','get-public-access-block'),('versioning','get-bucket-versioning'),('encryption','get-bucket-encryption'),('location','get-bucket-location'),('ownership','get-bucket-ownership-controls')]}
    (STATE.parent/'bucket_verification.json').write_text(json.dumps(checks,indent=2))
    return state


def put_immutable(path:Path,key:str,state:dict,expected_sha:str|None=None)->dict:
    digest=sha(path)
    if expected_sha and digest!=expected_sha:raise ValueError(f'Input SHA256 changed: {path}')
    checksum=base64.b64encode(bytes.fromhex(digest)).decode();size=path.stat().st_size
    head=aws('s3api','head-object','--bucket',state['bucket'],'--key',key,'--checksum-mode','ENABLED',allow_missing=True)
    if not head:
        start=time.perf_counter();response=aws('s3api','put-object','--bucket',state['bucket'],'--key',key,'--body',str(path),'--if-none-match','*','--checksum-algorithm','SHA256','--checksum-sha256',checksum,'--metadata',json.dumps({'sha256':digest}),'--server-side-encryption','AES256')
        head=aws('s3api','head-object','--bucket',state['bucket'],'--key',key,'--checksum-mode','ENABLED')
        seconds=time.perf_counter()-start
    else:seconds=0.
    if head['ContentLength']!=size or head.get('ChecksumSHA256')!=checksum or head.get('Metadata',{}).get('sha256')!=digest:raise ValueError(f'S3 verification mismatch {key}')
    result={'path':str(path),'key':key,'bytes':size,'sha256':digest,'s3_checksum_sha256':head['ChecksumSHA256'],'version_id':head.get('VersionId'),'upload_seconds':seconds}
    # One atomic receipt per key avoids losing completed uploads on interruption.
    receipts=STATE.parent/'receipts';receipts.mkdir(exist_ok=True);(receipts/(hashlib.sha256(key.encode()).hexdigest()+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({'verified_key':key,'bytes':size,'seconds':round(seconds,2)}),flush=True)
    return result


def raw_manifest()->dict:
    audit=json.loads(Path('docs/audit_evidence/audit.json').read_text());records=[]
    for path in sorted(Path('student_resource/dataset').glob('*/*.tsv')):
        name=path.stem;metadata=audit['files'].get(name)
        if metadata:rows=metadata['n'];schema=metadata['schema'];expected=metadata['sha256']
        elif name=='train_ground_truth':rows=audit['ground_truth']['integrity']['record_count'];schema={'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'};expected=audit['ground_truth']['sha256']
        else:raise ValueError('Unrecognized supplied raw file '+str(path))
        digest=sha(path)
        if digest!=expected:raise ValueError('Raw checksum differs from audited original')
        records.append({'path':str(path),'relative_path':str(path.relative_to('student_resource/dataset')),'bytes':path.stat().st_size,'sha256':digest,'rows':rows,'schema':schema,'mtime_ns':path.stat().st_mtime_ns})
    return {'version':'raw-v001','created_at':datetime.now(timezone.utc).isoformat(),'files':records,'total_bytes':sum(r['bytes'] for r in records),'source':'original supplied competition TSVs; exact byte copies','policy':'raw conditional creation + delete denial; administrator can change policy; not regulatory Object Lock'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','processed']);p.add_argument('--processed-root',type=Path,default=Path('artifacts/processed/prep-v001'));a=p.parse_args()
    state=setup()
    if a.mode=='raw':
        manifest_path=Path('manifests/MANIFEST_RAW.json')
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else raw_manifest()
        manifest_path.parent.mkdir(exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2))
        jobs=[(Path(r['path']),PREFIX+'raw/'+r['relative_path'],r['sha256']) for r in manifest['files']]
    else:
        manifest_path=a.processed_root/'manifest.json'
        if not manifest_path.exists():raise FileNotFoundError('Processed completion manifest required')
        jobs=[(path,PREFIX+'processed/prep-v001/'+str(path.relative_to(a.processed_root)),None) for path in sorted(a.processed_root.rglob('*')) if path.is_file() and path.suffix in ['.parquet','.json']]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda job:put_immutable(job[0],job[1],state,job[2]),jobs))
    if a.mode=='raw':results.append(put_immutable(manifest_path,PREFIX+'manifests/raw-v001/MANIFEST_RAW.json',state))
    output={'mode':a.mode,'bucket':state['bucket'],'region':REGION,'completed_at':datetime.now(timezone.utc).isoformat(),'objects':results,'total_bytes':sum(r['bytes'] for r in results)}
    (STATE.parent/(a.mode+'_upload.json')).write_text(json.dumps(output,indent=2));print(json.dumps({'mode':a.mode,'objects':len(results),'total_bytes':output['total_bytes']}))

if __name__=='__main__':main()
