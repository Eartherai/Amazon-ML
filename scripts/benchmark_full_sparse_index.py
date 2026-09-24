"""Persist and benchmark one full-country lexical index without consulting labels."""
from pathlib import Path
import gc,json,time,hashlib,shutil,resource
import duckdb,numpy as np,polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.char_retrieval import fused_top_k_rows
out=Path('artifacts/retrieval/P4-INDEX-PROBE-001');out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
if shutil.disk_usage('.').free<10*1024**3:raise RuntimeError('Need10GiB free before index probe')
packed=np.load('outputs/candidates/P4-A-001/name_char3_idf.npz',allow_pickle=True);v=TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=False,vocabulary={w:i for i,w in enumerate(packed['vocabulary'])},dtype=np.float32);v.fit(['placeholder']);v.idf_=packed['idf']
db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'});cur=db.execute("SELECT entity_id,n FROM targets_normalized WHERE country='India'");parts=[];idparts=[]
while rows:=cur.fetchmany(25000):parts.append(v.transform([r[1] for r in rows]));idparts.append(np.array([r[0] for r in rows]))
index=sparse.vstack(parts,format='csr');ids=np.concatenate(idparts);del parts,idparts;gc.collect();build=time.perf_counter()-start;sparse.save_npz(out/'name-India.npz',index,compressed=True);np.save(out/'name-India-ids.npy',ids,allow_pickle=False);store=time.perf_counter()-start-build
q=pl.read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet').filter(pl.col('country')=='India').sort('entity_id').head(1000);matrix=v.transform(q['n'].to_list());target=index.T.tocsr();begin=time.perf_counter();results=fused_top_k_rows(matrix,target,ids,100,threads=2,transposed=True);seconds=time.perf_counter()-begin
actual={qid:set(ii) for qid,(ii,_) in zip(q['entity_id'],results)};ref=pl.read_parquet('outputs/candidates/P4-A-001/name_char3.parquet');expected={qid:set() for qid in actual}
for qid,t in ref.select('source1_entity_id','target_id').iter_rows():
 if qid in expected:expected[qid].add(t)
different=sum(actual[qid]!=expected[qid] for qid in actual);report={'country':'India','field':'name','target_rows':len(ids),'query_count':len(q),'build_seconds':build,'save_seconds':store,'search_seconds':seconds,'different_candidate_sets_vs_reference':different,'csr_bytes':index.data.nbytes+index.indices.nbytes+index.indptr.nbytes,'stored_bytes':sum(p.stat().st_size for p in out.glob('*')),'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'idf_sha256':hashlib.sha256(Path('outputs/candidates/P4-A-001/name_char3_idf.npz').read_bytes()).hexdigest(),'scope':'Full-country index,1000 queries; labels never read. Cached text index, not a new model.'};(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
