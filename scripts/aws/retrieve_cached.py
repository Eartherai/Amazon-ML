"""Stream query batches through persistent indexes; write bounded route shards.

No labels are read. Query-shard ownership is SHA256(entity ID) modulo shard count.
Every query, including zero-candidate records, is retained in a separate coverage file.
"""
import argparse,hashlib,json,time,tarfile
from pathlib import Path
import numpy as np,polars as pl
from scipy import sparse
from benchmark_indexes import vectorizer
from src.blocking.char_retrieval import fused_top_k_rows,top_k


def shard_id(entity_id,count):
    return int.from_bytes(hashlib.sha256(entity_id.encode('utf-8')).digest()[:8],'big')%count


def main():
    p=argparse.ArgumentParser();p.add_argument('--indexes',type=Path,required=True);p.add_argument('--idf',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1000);p.add_argument('--threads',type=int,default=8);p.add_argument('--shards',type=int,default=64);p.add_argument('--upload-bucket');p.add_argument('--upload-prefix');a=p.parse_args()
    if min(a.batch_size,a.threads,a.shards)<1:raise ValueError('Positive sizes required')
    if bool(a.upload_bucket)!=bool(a.upload_prefix):raise ValueError('Both upload settings required')
    a.output.mkdir(parents=True,exist_ok=False)
    qs=pl.read_parquet(a.queries,columns=['entity_id','country','n','a'])
    if qs['entity_id'].n_unique()!=len(qs):raise ValueError('Duplicate S1 IDs')
    qs=qs.with_columns(pl.Series('shard',[shard_id(q,a.shards) for q in qs['entity_id']],dtype=pl.UInt16))
    qs.select('entity_id','country','shard').write_parquet(a.output/'query_coverage.parquet',compression='zstd')
    index_report=json.loads((a.indexes/'metrics.json').read_text())
    coverage={(r['field'],r['country']) for r in index_report['routes']}
    for country in qs['country'].unique():
        for field in ['name','address']:
            if (field,country) not in coverage:raise ValueError(f'Missing country index: {country}/{field}')
    report={'query_count':len(qs),'shard_algorithm':'SHA256 UTF8 first8bytes big endian modulo N','shards':a.shards,'fold4_labels':'NOT READ','routes':[]};start=time.perf_counter()
    for field,col in [('name','n'),('address','a')]:
        v=vectorizer(a.idf/f'{field}_char3_idf.npz')
        for ci,country in enumerate(sorted(qs['country'].unique().to_list())):
            # Resolve actual directory from its stored country, not enumeration assumptions.
            directories=[d for d in a.indexes.glob(field+'-country-*') if json.loads((d/'manifest.json').read_text())['country']==country]
            if len(directories)!=1:raise ValueError('Ambiguous/missing index')
            directory=directories[0];manifest=json.loads((directory/'manifest.json').read_text())
            indexes=[(sparse.load_npz(directory/f'{s}.npz'),np.load(directory/f'{s}-ids.npy',allow_pickle=False)) for s in manifest['shards']]
            country_q=qs.filter(pl.col('country')==country);route_count=0;begin=time.perf_counter()
            for shard in range(a.shards):
                group=country_q.filter(pl.col('shard')==shard).sort('entity_id')
                for batch_no,lo in enumerate(range(0,len(group),a.batch_size)):
                    batch=group.slice(lo,a.batch_size);matrix=v.transform(batch[col].to_list());best=[(np.array([],dtype='U32'),np.array([],dtype=np.float32)) for _ in range(len(batch))]
                    for target,ids in indexes:
                        selected=fused_top_k_rows(matrix,target,ids,100,threads=a.threads,transposed=True)
                        for i,(ii,ss) in enumerate(selected):best[i]=top_k(np.concatenate([best[i][0],ii]),np.concatenate([best[i][1],ss]),100)
                    rows=[(q,str(t),float(s),rank+1) for q,(ids,scores) in zip(batch['entity_id'],best) for rank,(t,s) in enumerate(zip(ids,scores))]
                    pl.DataFrame(rows,schema={'source1_entity_id':pl.String,'target_id':pl.String,'route_score':pl.Float32,'route_rank':pl.UInt16},orient='row').write_parquet(a.output/f'{field}-c{ci:03d}-s{shard:03d}-b{batch_no:05d}.parquet',compression='zstd')
                    route_count+=len(rows)
                if a.upload_bucket and len(group):
                    from upload_verified import upload
                    archive=a.output/f'{field}-c{ci:03d}-s{shard:03d}.tar'
                    with tarfile.open(archive,'w') as tar:
                        for part in sorted(a.output.glob(f'{field}-c{ci:03d}-s{shard:03d}-b*.parquet')):tar.add(part,arcname=part.name,recursive=False)
                    receipt=upload(archive,a.upload_bucket,a.upload_prefix.rstrip('/')+'/'+archive.name)
                    (a.output/(archive.stem+'-receipt.json')).write_text(json.dumps(receipt,indent=2))
                    archive.unlink()  # Verified remote archive retained; local Parquet stays intact.
                print(json.dumps({'field':field,'country':country,'shard':shard,'pairs_so_far':route_count,'seconds':time.perf_counter()-begin}),flush=True)
            report['routes'].append({'field':field,'country':country,'queries':len(country_q),'pairs':route_count,'seconds':time.perf_counter()-begin});report['seconds']=time.perf_counter()-start
            (a.output/'progress.json').write_text(json.dumps(report,indent=2));del indexes
    (a.output/'COMPLETE.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
