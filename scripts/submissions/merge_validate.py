"""Merge sorted frozen SUB-001 shard files and run strict + official validation."""
import argparse,gzip,hashlib,heapq,json,subprocess,time
from pathlib import Path

HEADERS={'matching':'source1_entity_id\tmatched_entity_ids\n','candidates':'source1_entity_id\tcandidate_entity_ids\n'}

def lines(path,header):
    with gzip.open(path,'rt',encoding='utf-8') as source:
        if source.readline()!=header:raise ValueError(f'Wrong shard header: {path}')
        last=''
        for line in source:
            q,tab,_=line.partition('\t')
            if not tab or (last and q<=last):raise ValueError(f'Shard not strictly sorted: {path}')
            last=q;yield line

def merged(kind,shards,target):
    files=sorted(shards.glob(f'*-{kind}.tsv.gz'))
    if not files:raise ValueError('No shards for '+kind)
    with target.open('w',encoding='utf-8',newline='') as dest:
        dest.write(HEADERS[kind]);last='';count=0
        for row in heapq.merge(*(lines(path,HEADERS[kind]) for path in files)):
            q,_,_=row.partition('\t')
            if last and q<=last:raise ValueError('Duplicate or unsorted S1 across shards')
            dest.write(row);last=q;count+=1
    return count,len(files)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--shards',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--test-dir',type=Path,required=True);p.add_argument('--expected',type=int,default=1732544);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    matching=a.output/'matching_results.tsv';candidate=a.output/'candidate_pairs.tsv'
    mc,mfiles=merged('matching',a.shards,matching);cc,cfiles=merged('candidates',a.shards,candidate)
    if mc!=a.expected or cc!=a.expected:raise ValueError(f'Incomplete: matching {mc} candidate {cc} expected {a.expected}')
    with matching.open(encoding='utf-8')as m,candidate.open(encoding='utf-8')as c:
        next(m);next(c)
        for ml,cl in zip(m,c,strict=True):
            mq,_,matches=ml.rstrip('\n').partition('\t');cq,_,candidates=cl.rstrip('\n').partition('\t')
            if mq!=cq:raise ValueError('S1 alignment mismatch')
            mt=matches.split(',')if matches else [];ct=candidates.split(',')if candidates else []
            if len(mt)!=len(set(mt)) or len(ct)!=len(set(ct)) or not set(mt)<=set(ct):raise ValueError('Duplicate ID or match outside scored candidates')
            if any(not t.startswith(('S2-','S3-')) for t in ct):raise ValueError('Invalid target prefix')
    # The exact official command is executed from student_resource, including heavy ID validation.
    project=Path.cwd();student=project/'student_resource';link=student/'output'
    if not link.exists():link.symlink_to(a.output.resolve(),target_is_directory=True)
    elif link.resolve()!=a.output.resolve():raise ValueError('student_resource/output points elsewhere')
    if a.test_dir.resolve()!=(student/'dataset/test').resolve():raise ValueError('Expected original test directory')
    cmd=['python3','utils/validate_submission.py','--matching','output/matching_results.tsv','--candidate','output/candidate_pairs.tsv','--test-dir','dataset/test']
    validation={}
    for name,args in [('official',cmd),('official_check_ids',cmd+['--check-ids'])]:
        run=subprocess.run(args,cwd=student,capture_output=True,text=True)
        validation[name]={'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        (a.output/(name+'.log')).write_text(run.stdout+'\n'+run.stderr)
        if run.returncode!=0 or 'PASS' not in run.stdout:raise RuntimeError(f'{name} failed; inspect log')
    result={'rows':mc,'matching_shards':mfiles,'candidate_shards':cfiles,'matching_sha256':sha(matching),'candidate_sha256':sha(candidate),'model':'SUB-001 frozen LightGBM NUMERIC-V2','threshold':0.83,'validation':{k:{'exit_code':v['exit_code'],'pass':'PASS' in v['stdout']}for k,v in validation.items()},'seconds':time.perf_counter()-started}
    (a.output/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
