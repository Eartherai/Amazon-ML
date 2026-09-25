"""SUB-002: same test retrieval/features with a larger fitted model and score sidecars.

Every candidate in the shard output is scored. Inputs are only competition test records
and the frozen development model/IDF. Countries are enumerated from the test data.
"""
import argparse,gzip,hashlib,json,shutil,time,sys
from pathlib import Path
import duckdb,lightgbm as lgb,numpy as np,polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.char_retrieval import fused_top_k_rows,top_k
from src.models.sub001_features import NAMES,pair_features


def shard_id(value,count):return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'big')%count

def frozen_vectorizer(path):
    packed=np.load(path,allow_pickle=True) # Trusted, SHA256-checked frozen project artifact.
    vector=TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=False,vocabulary={word:i for i,word in enumerate(packed['vocabulary'])},dtype=np.float32)
    vector.fit(['placeholder']);vector.idf_=packed['idf'];return vector


def indexes(con,country,col,vector,chunk_rows):
    """Keep all country targets indexed once, with stable lexical ID ordering."""
    cursor=con.execute(f'SELECT entity_id,{col} FROM targets WHERE country=? ORDER BY entity_id',[country]);parts=[];target_lookup={}
    while rows:=cursor.fetchmany(chunk_rows):
        ids=np.array([r[0]for r in rows]);matrix=vector.transform([r[1]for r in rows]).T.tocsr();parts.append((matrix,ids))
    return parts


def retrieve(matrix,parts,k,threads):
    best=[(np.array([],dtype='U32'),np.array([],dtype=np.float32))for _ in range(matrix.shape[0])]
    for target,ids in parts:
        selected=fused_top_k_rows(matrix,target,ids,k,threads=threads,transposed=True)
        for i,(ii,ss) in enumerate(selected):
            best[i]=top_k(np.concatenate((best[i][0],ii)),np.concatenate((best[i][1],ss)),k)
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=int,default=64);p.add_argument('--batch-size',type=int,default=200);p.add_argument('--threads',type=int,default=8);p.add_argument('--max-queries-per-country',type=int);p.add_argument('--country');p.add_argument('--first-shard',type=int,default=0);p.add_argument('--last-shard',type=int);p.add_argument('--upload-bucket');p.add_argument('--upload-prefix');p.add_argument('--score-floor',type=float,default=0.4);p.add_argument('--top-k',type=int,default=2,help='sidecar keeps score>=floor or top-k (SUB-002 used 2)');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.shards<1 or a.batch_size<1 or a.threads<1 or not 0<=a.score_floor<=1:raise ValueError('Invalid sizes or score floor')
    if a.last_shard is None:a.last_shard=a.shards
    if not 0<=a.first_shard<a.last_shard<=a.shards:raise ValueError('Invalid shard interval')
    cfg=json.loads((a.inputs/'submission-config.json').read_text());train=json.loads((a.inputs/'train-manifest.json').read_text())
    if cfg['threshold']!=train['threshold'] or train['feature_names']!=NAMES or train['fold4']!='CLOSED':raise ValueError('Frozen model/config mismatch')
    a.output.mkdir(parents=True);(a.output/'shards').mkdir();start=time.perf_counter()
    vectors={field:frozen_vectorizer(a.inputs/f'{field}_char3_idf.npz')for field in ['name','address']}
    model=lgb.Booster(model_file=str(a.inputs/'model.txt'))
    if model.feature_name()!=NAMES:raise ValueError('Model feature names/order changed')
    trans=pl.read_parquet(a.inputs/'name_map.parquet');transmap=dict(zip(trans['n'],trans['transliterated']))
    con=duckdb.connect(config={'threads':4,'memory_limit':'4GB'})
    con.read_parquet([str(a.inputs/'target2.parquet'),str(a.inputs/'target3.parquet')]).create_view('targets')
    queries=pl.read_parquet(a.inputs/'queries.parquet')
    if queries['entity_id'].n_unique()!=len(queries):raise ValueError('Duplicate test S1')
    query_total=len(queries)
    queries=queries.with_columns(pl.Series('shard',[shard_id(x,a.shards)for x in queries['entity_id']],dtype=pl.UInt16))
    countries=sorted(queries['country'].unique().to_list())
    if a.country:
        if a.country not in countries:raise ValueError(f'Unknown query country: {a.country}')
        countries=[a.country]
    report={'config':cfg,'model_sha256':train['model_sha256'],'query_total':query_total,'processed_query_count':0,'countries':countries,'fold4':'CLOSED','country_progress':[],'started_seconds':start}
    for country in countries:
        cq=queries.filter(pl.col('country')==country)
        if a.max_queries_per_country:cq=cq.sort('entity_id').head(a.max_queries_per_country)
        cq=cq.filter((pl.col('shard')>=a.first_shard)&(pl.col('shard')<a.last_shard))
        # Target metadata is kept once per country. None of these values is an external lookup.
        lookup={qid:(name,address)for qid,name,address in con.execute('SELECT entity_id,n,a FROM targets WHERE country=?',[country]).fetchall()}
        routes={field:indexes(con,country,col,vectors[field],250000)for field,col in [('name','n'),('address','a')]}
        country_stats={'country':country,'queries':len(cq),'targets':len(lookup),'pairs':0,'matches':0,'shards':0,'seconds':0};country_start=time.perf_counter()
        for shard in range(a.first_shard,a.last_shard):
            group=cq.filter(pl.col('shard')==shard).sort('entity_id')
            if not len(group):continue
            base=f'{country.replace("/","_")}-s{shard:03d}';cp=a.output/'shards'/f'{base}-candidates.tsv.gz';mp=a.output/'shards'/f'{base}-matching.tsv.gz';sp=a.output/'shards'/f'{base}-scores.tsv.gz'
            with gzip.open(cp,'wt',encoding='utf-8',newline='') as candidate_file,gzip.open(mp,'wt',encoding='utf-8',newline='') as matching_file,gzip.open(sp,'wt',encoding='utf-8',newline='') as score_file:
                candidate_file.write('source1_entity_id\tcandidate_entity_ids\n');matching_file.write('source1_entity_id\tmatched_entity_ids\n');score_file.write('source1_entity_id\ttarget_id\tscore\n')
                for lo in range(0,len(group),a.batch_size):
                    batch=group.slice(lo,a.batch_size);retrieved={}
                    for field,col in [('name','n'),('address','a')]:
                        matrix=vectors[field].transform(batch[col].to_list());retrieved[field]=retrieve(matrix,routes[field],100,a.threads)
                    features=[];pairids=[];owners=[];candidates=[]
                    for i,(qid,_,qn,qa,_) in enumerate(batch.select('entity_id','country','n','a','shard').iter_rows()):
                        scores={}
                        for field in ['name','address']:
                            ids,vals=retrieved[field][i]
                            for rank,(tid,score) in enumerate(zip(ids,vals),start=1):
                                key=str(tid);slot=scores.setdefault(key,[0.,0.,0.,0.,0])
                                offset=0 if field=='name' else 1;slot[offset]=float(score);slot[2+offset]=1./rank;slot[4]+=1
                        ordered=sorted(scores);candidates.append((qid,ordered));
                        for tid in ordered:
                            tn,ta=lookup[tid];ns,ads,nr,ar,rc=scores[tid]
                            features.append(pair_features(qn,qa,country,tn,ta,country,tid,ns,ads,nr,ar,rc,transmap));pairids.append(tid);owners.append(qid)
                    probabilities=model.predict(np.stack(features),num_threads=a.threads) if features else np.empty(0)
                    selected={qid:[]for qid,_ in candidates};ranked={qid:[]for qid,_ in candidates}
                    for qid,tid,score in zip(owners,pairids,probabilities):
                        ranked[qid].append((tid,float(score)))
                        if score>=cfg['threshold']:selected[qid].append(tid)
                    for qid,allids in candidates:
                        chosen=sorted(set(selected[qid]));assert set(chosen)<=set(allids)
                        top_two={tid for tid,_ in sorted(ranked[qid],key=lambda item:(-item[1],item[0]))[:a.top_k]}
                        for tid,score in ranked[qid]:
                            if score>=a.score_floor or tid in top_two:score_file.write(qid+'\t'+tid+'\t'+format(score,'.9g')+'\n')
                        candidate_file.write(qid+'\t'+','.join(allids)+'\n');matching_file.write(qid+'\t'+','.join(chosen)+'\n')
                        country_stats['pairs']+=len(allids);country_stats['matches']+=len(chosen)
            if a.upload_bucket:
                sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
                from scripts.aws.upload_verified import upload
                receipts=[upload(path,a.upload_bucket,a.upload_prefix.rstrip('/')+('/scores/' if path==sp else '/shards/')+path.name)for path in (cp,mp,sp)]
                (a.output/'shards'/f'{base}-receipts.json').write_text(json.dumps(receipts,indent=2))
            country_stats['shards']+=1;country_stats['seconds']=time.perf_counter()-country_start
            (a.output/'progress.json').write_text(json.dumps({**report,'active':country_stats},indent=2))
            print(json.dumps({'country':country,'shard':shard,'query_rows':len(group),'pairs_so_far':country_stats['pairs'],'seconds':country_stats['seconds']}),flush=True)
        report['country_progress'].append(country_stats);report['processed_query_count']+=len(cq);(a.output/'progress.json').write_text(json.dumps(report,indent=2))
        del routes,lookup
    report['total_seconds']=time.perf_counter()-start
    (a.output/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({'complete':True,'seconds':report['total_seconds'],'queries':report['processed_query_count']}),flush=True)
if __name__=='__main__':main()
