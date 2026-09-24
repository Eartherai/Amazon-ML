"""Build reusable country/source char3 shards and benchmark exact merged top100.

Inputs contain no labels. Each immutable index stores CSR transpose and record IDs.
Country values are enumerated, never hard-coded. Matches the frozen IDF pipeline.
"""
import argparse,gc,hashlib,json,resource,time
from pathlib import Path
import duckdb,numpy as np,polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.char_retrieval import fused_top_k_rows,top_k


def vectorizer(path):
    packed=np.load(path,allow_pickle=True)  # Trusted frozen project artifact, SHA256-verified before this job.
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=False,vocabulary={w:i for i,w in enumerate(packed['vocabulary'])},dtype=np.float32)
    v.fit(['placeholder']);v.idf_=packed['idf']
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--queries-per-country',type=int,default=200);p.add_argument('--chunk-rows',type=int,default=250000);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    con=duckdb.connect(config={'threads':4,'memory_limit':'8GB'})
    con.read_parquet(str(a.input/'targets.parquet')).project("*,substr(entity_id,1,2) AS source").create_view('targets')
    queries=pl.read_parquet(a.input/'queries.parquet')
    report={'benchmark_only':True,'fold4':'CLOSED','input_manifest':json.loads((a.input/'manifest.json').read_text()),'routes':[]}
    for field,col in [('name','n'),('address','a')]:
        v=vectorizer(a.input/f'{field}_char3_idf.npz');ref=pl.read_parquet(a.input/f'{field}_char3.parquet')
        for ci,(country,) in enumerate(con.execute('SELECT DISTINCT country FROM targets ORDER BY country').fetchall()):
            path=a.output/f'{field}-country-{ci:03d}';path.mkdir()
            qs=queries.filter(pl.col('country')==country).sort('entity_id').head(a.queries_per_country)
            matrix=v.transform(qs[col].to_list());shards=[];build=time.perf_counter()
            for source, in con.execute('SELECT DISTINCT source FROM targets WHERE country=? ORDER BY source',[country]).fetchall():
                con.execute('SELECT entity_id,'+col+' FROM targets WHERE country=? AND source=? ORDER BY entity_id',[country,source])
                j=0
                while rows:=con.fetchmany(a.chunk_rows):
                    ids=np.array([r[0] for r in rows]);target=v.transform([r[1] for r in rows]).T.tocsr()
                    stem=f'{source}-{j:04d}';sparse.save_npz(path/f'{stem}.npz',target,compressed=False);np.save(path/f'{stem}-ids.npy',ids,allow_pickle=False)
                    shards.append(stem);j+=1
            del rows,target,ids;gc.collect()
            built=time.perf_counter()-build
            begin=time.perf_counter();loaded=[(sparse.load_npz(path/f'{s}.npz'),np.load(path/f'{s}-ids.npy',allow_pickle=False)) for s in shards];load=time.perf_counter()-begin
            expected={q:set() for q in qs['entity_id']}
            for q,t in ref.select('source1_entity_id','target_id').iter_rows():
                if q in expected:expected[q].add(t)
            trials=[]
            for threads in [2,4,8]:
                start=time.perf_counter();best=[(np.array([],dtype='U32'),np.array([],dtype=np.float32)) for _ in range(len(qs))]
                for target,ids in loaded:
                    selected=fused_top_k_rows(matrix,target,ids,100,threads=threads,transposed=True)
                    for i,(ii,ss) in enumerate(selected):best[i]=top_k(np.concatenate([best[i][0],ii]),np.concatenate([best[i][1],ss]),100)
                elapsed=time.perf_counter()-start
                mismatches=sum(set(ii)!=expected[q] for q,(ii,_) in zip(qs['entity_id'],best))
                trials.append({'threads':threads,'seconds':elapsed,'queries_per_second':len(qs)/elapsed,'candidate_set_mismatches':mismatches})
                if mismatches:raise RuntimeError(f'Parity mismatch: {field} {country}: {mismatches}')
            entry={'field':field,'country':country,'source_partitioned':True,'shards':shards,'queries':len(qs),'build_save_seconds':built,'load_seconds':load,'bytes':sum(x.stat().st_size for x in path.iterdir()),'trials':trials}
            (path/'manifest.json').write_text(json.dumps(entry,indent=2));report['routes'].append(entry);report['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
            (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(entry),flush=True)
            del loaded;gc.collect()

if __name__=='__main__':main()
