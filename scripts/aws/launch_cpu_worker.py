"""Launch the frozen bounded EC2 index benchmark; fail before launch if unready."""
import argparse,hashlib,json,subprocess,shlex
from datetime import datetime,timezone
from pathlib import Path
from provision_worker import aws
from upload_verified import upload
root=Path('artifacts/cloud/phase5')

def s3_keys(bucket:str,prefix:str)->list[str]:
 """List every object page explicitly before checking a completed run."""
 keys=[];token=None
 while True:
  command=['s3api','list-objects-v2','--no-paginate','--bucket',bucket,'--prefix',prefix]
  if token:command.extend(['--continuation-token',token])
  page=aws(*command)
  keys.extend(row['Key'] for row in page.get('Contents',[]))
  if not page.get('IsTruncated'):return keys
  token=page.get('NextContinuationToken')
  if not token:raise RuntimeError('S3 inventory truncated without continuation token')

parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/aws/P5-INDEX-001.json'));args=parser.parse_args()
cfg=json.loads(args.config.read_text());infra=json.loads((root/'infrastructure.json').read_text())
from cost_guard import guard
if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Commit exact worker code before launch')
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
account=aws('sts','get-caller-identity')['Account']
if account[-4:]!=infra['account_suffix']:raise RuntimeError('Account mismatch')
quota=aws('service-quotas','get-service-quota','--service-code','ec2','--quota-code',('L-34B43A08' if cfg.get('purchase')=='spot' else 'L-1216C47A'))['Quota']['Value']
if quota<8:raise RuntimeError('Insufficient quota')
if not (root/cfg.get('input_receipt','input-upload.json')).exists():raise RuntimeError('Inputs not verified')
if cfg.get('job_kind') in {'full_retrieval','sample_retrieval'}:
 benchmark=json.loads((root/'P5-INDEX-001/ledger.json').read_text())
 if benchmark.get('exit_code')!=0 or benchmark.get('instance_state')!='terminated':raise RuntimeError('Benchmark must succeed and terminate first')
if cfg.get('job_kind')=='sample_features':
 retrieval=json.loads((root/'P5-LEARNING-200K-001/ledger.json').read_text())
 if retrieval.get('exit_code')!=0 or retrieval.get('instance_state')!='terminated':raise RuntimeError('Complete 200k retrieval must terminate successfully first')
 complete=root/'P5-LEARNING-200K-001/results-COMPLETE.json'
 if not complete.exists() or json.loads(complete.read_text()).get('query_count')!=200000:raise RuntimeError('Missing complete 200k retrieval summary')
 archive_prefix=cfg['shard_prefix'].rstrip('/')+'/shards/'
 archives=[key for key in s3_keys(infra['bucket'],archive_prefix) if key.endswith('.tar')]
 if len(archives)!=256:raise RuntimeError('Incomplete 256-archive retrieval checkpoint')
if cfg.get('job_kind')=='learning_curve':
 features=json.loads((root/'P5-FEATURE-200K-001/ledger.json').read_text())
 if features.get('exit_code')!=0 or features.get('instance_state')!='terminated':raise RuntimeError('Complete 200k feature job must terminate successfully first')
 complete=root/'P5-FEATURE-200K-001/results-COMPLETE.json'
 if not complete.exists():raise RuntimeError('Missing complete feature summary')
 expected_parts=json.loads(complete.read_text())['feature_parts']
 feature_prefix=cfg['shard_prefix'].rstrip('/')+'/results/'
 feature_parts=[key for key in s3_keys(infra['bucket'],feature_prefix) if key.split('/')[-1].startswith('features-') and key.endswith('.parquet')]
 if len(feature_parts)!=expected_parts:raise RuntimeError('Incomplete feature part checkpoint')
price_cap=str(cfg.get('spot_max_price',cfg['compute_usd_per_hour']))
cost_record=guard(cfg,price_cap)
run=root/cfg['run_id'];run.mkdir(exist_ok=False)
(run/'cost_guard.json').write_text(json.dumps(cost_record,indent=2))
archive=run/'code.tar';subprocess.run(['git','archive','--format=tar','--output',str(archive),commit],check=True)
key='amazon-ml-2026/phase5/code/'+commit+'.tar';receipt=upload(archive,infra['bucket'],key)
ami=aws('ssm','get-parameter','--name','/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64')['Parameter']['Value']
offerings=aws('ec2','describe-instance-type-offerings','--location-type','availability-zone','--filters','Name=instance-type,Values='+cfg['instance_type'])['InstanceTypeOfferings']
azs={x['Location'] for x in offerings};subnet=next(s for s in infra['subnets'] if s['az'] in azs and (not cfg.get('availability_zone') or s['az']==cfg['availability_zone']))
values={'JOB_KIND':cfg.get('job_kind','index_benchmark'),'COUNTRY':cfg.get('country',''),'FIRST_SHARD':str(cfg.get('first_shard',0)),'LAST_SHARD':str(cfg.get('last_shard',64)),'SHUTDOWN_MINUTES':str(cfg['runtime_cap_minutes']),'INDEX_PREFIX':cfg.get('index_prefix',''),'SHARD_PREFIX':cfg.get('shard_prefix',''),'BUCKET':infra['bucket'],'CODE_KEY':key,'CODE_SHA256':receipt['sha256'],'INPUT_PREFIX':cfg['input_prefix'],'OUTPUT_PREFIX':'amazon-ml-2026/phase5/runs/'+cfg['run_id']}
script=Path('scripts/aws/worker_bootstrap.sh').read_text().splitlines()
if cfg.get('bootstrap_script'):
 script=Path(cfg['bootstrap_script']).read_text().splitlines()
bootstrap='\n'.join([script[0],*[k+'='+shlex.quote(v) for k,v in values.items()],*script[1:]])+'\n'
(run/'user-data.sh').write_text(bootstrap)
request={'ImageId':ami,'InstanceType':cfg['instance_type'],'MinCount':1,'MaxCount':1,'ClientToken':cfg['run_id'], 'IamInstanceProfile':{'Name':infra['instance_profile']},'InstanceInitiatedShutdownBehavior':'terminate','MetadataOptions':{'HttpTokens':'required','HttpPutResponseHopLimit':1},'NetworkInterfaces':[{'DeviceIndex':0,'SubnetId':subnet['id'],'Groups':[infra['security_group']],'AssociatePublicIpAddress':True}],'BlockDeviceMappings':[{'DeviceName':'/dev/xvda','Ebs':{'VolumeSize':cfg['ebs_gib'],'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],'TagSpecifications':[{'ResourceType':t,'Tags':[{'Key':'Project','Value':'aml2026-phase5'},{'Key':'RunId','Value':cfg['run_id']}]} for t in ['instance','volume']]}
if cfg.get('purchase')=='spot':
 request['InstanceMarketOptions']={'MarketType':'spot','SpotOptions':{'SpotInstanceType':'one-time','InstanceInterruptionBehavior':'terminate','MaxPrice':str(cfg['spot_max_price'])}}
(run/'request.json').write_text(json.dumps(request,indent=2))
ledger={**cfg,'git_sha':commit,'account_suffix':account[-4:],'start':datetime.now(timezone.utc).isoformat(),'stop':None,'actual_cost':None,'s3_inputs':cfg['input_prefix'],'s3_outputs':values['OUTPUT_PREFIX'],'code_upload':receipt}
(run/'ledger.json').write_text(json.dumps(ledger,indent=2))
response=aws('ec2','run-instances','--cli-input-json',json.dumps(request),'--user-data','file://'+str(run/'user-data.sh'))
ledger['instance_id']=response['Instances'][0]['InstanceId'];(run/'ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
