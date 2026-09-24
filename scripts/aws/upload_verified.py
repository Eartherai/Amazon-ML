"""Upload local outputs with conditional creation and SHA256 verification via CLI."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path

def cli(*args):
 r=subprocess.run(['aws','--region','us-east-1','--no-cli-pager',*args,'--output','json'],capture_output=True,text=True,check=True)
 return json.loads(r.stdout) if r.stdout.strip() else {}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def upload(path,bucket,key):
 digest=sha(path);checksum=base64.b64encode(bytes.fromhex(digest)).decode()
 cli('s3api','put-object','--bucket',bucket,'--key',key,'--body',str(path),'--if-none-match','*','--checksum-algorithm','SHA256','--checksum-sha256',checksum)
 head=cli('s3api','head-object','--bucket',bucket,'--key',key,'--checksum-mode','ENABLED')
 if head['ChecksumSHA256']!=checksum or head['ContentLength']!=path.stat().st_size:raise RuntimeError('Upload verification failed')
 return {'key':key,'sha256':digest,'bytes':head['ContentLength'],'version_id':head.get('VersionId')}
if __name__=='__main__':
 root=Path(sys.argv[1]);bucket=sys.argv[2];prefix=sys.argv[3].rstrip('/')
 receipts=[]
 for p in sorted(root.rglob('*')):
  if p.is_file():
   receipts.append(upload(p,bucket,prefix+'/'+str(p.relative_to(root))));print(json.dumps(receipts[-1]),flush=True)
 Path(sys.argv[4]).write_text(json.dumps(receipts,indent=2))
