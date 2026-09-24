"""Create a dedicated EC2 S3-only role/profile and zero-ingress security group."""
import json,subprocess,os
from pathlib import Path
REGION='us-east-1';NAME='aml2026-phase5-worker'
def aws(*args):
 p=subprocess.run(['aws','--profile',os.environ.get('AWS_PROFILE','amamzon_01_a1_0'),'--region',REGION,'--no-cli-pager',*args,'--output','json'],text=True,capture_output=True,check=True)
 return json.loads(p.stdout) if p.stdout.strip() else {}
def main():
 state=json.loads(Path('artifacts/cloud/phase3/s3_state.json').read_text())
 identity=aws('sts','get-caller-identity')
 if identity['Account'][-4:]!=state['account_suffix']:raise RuntimeError('Wrong account')
 bucket=state['bucket'];base=f'arn:aws:s3:::{bucket}'
 trust={'Version':'2012-10-17','Statement':[{'Effect':'Allow','Principal':{'Service':'ec2.amazonaws.com'},'Action':'sts:AssumeRole'}]}
 policy={'Version':'2012-10-17','Statement':[
 {'Effect':'Allow','Action':['s3:GetObject'],'Resource':[base+'/amazon-ml-2026/phase5/inputs/*',base+'/amazon-ml-2026/phase5/code/*']},
 {'Effect':'Allow','Action':['s3:PutObject','s3:GetObject'],'Resource':base+'/amazon-ml-2026/phase5/runs/*'},
 {'Effect':'Allow','Action':['s3:ListBucket'],'Resource':base,'Condition':{'StringLike':{'s3:prefix':['amazon-ml-2026/phase5/inputs/*','amazon-ml-2026/phase5/code/*','amazon-ml-2026/phase5/runs/*']}}}]}
 Path('infra/worker-trust.json').write_text(json.dumps(trust,indent=2));Path('infra/worker-s3-policy.json').write_text(json.dumps(policy,indent=2))
 existing=aws('iam','list-roles','--query','Roles[].RoleName')
 if NAME in existing:raise RuntimeError('Existing role: inspect instead of overwriting')
 aws('iam','create-role','--role-name',NAME,'--assume-role-policy-document',json.dumps(trust),'--tags','Key=Project,Value=aml2026-phase5')
 aws('iam','put-role-policy','--role-name',NAME,'--policy-name','Phase5Artifacts','--policy-document',json.dumps(policy))
 aws('iam','create-instance-profile','--instance-profile-name',NAME)
 aws('iam','add-role-to-instance-profile','--instance-profile-name',NAME,'--role-name',NAME)
 subnets=aws('ec2','describe-subnets','--filters','Name=default-for-az,Values=true')['Subnets']
 if not subnets:raise RuntimeError('No default subnet')
 sg=aws('ec2','create-security-group','--group-name',NAME,'--description','Finite experiment worker: no inbound ports','--vpc-id',subnets[0]['VpcId'])['GroupId']
 config={'region':REGION,'account_suffix':state['account_suffix'],'bucket':bucket,'instance_profile':NAME,'security_group':sg,'subnets':[{'id':s['SubnetId'],'az':s['AvailabilityZone']} for s in subnets]}
 Path('artifacts/cloud/phase5/infrastructure.json').write_text(json.dumps(config,indent=2));print(json.dumps(config))
if __name__=='__main__':main()
