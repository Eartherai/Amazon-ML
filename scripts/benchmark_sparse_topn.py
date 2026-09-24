"""Check fused sparse top-K speed/parity on an unlabeled bounded text subset."""
from pathlib import Path
import json,time,importlib.metadata
import duckdb,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.blocking.char_retrieval import top_k
out=Path('outputs/benchmarks/P4-TOPN-001');out.mkdir(parents=True,exist_ok=False)
db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'1GB'});reports=[]
for field,col in [('name','n'),('address','a')]:
 packed=np.load(f'outputs/candidates/P4-A-001/{field}_char3_idf.npz',allow_pickle=True);vocab={v:i for i,v in enumerate(packed['vocabulary'])};v=TfidfVectorizer(analyzer='char',ngram_range=(3,3),lowercase=False,vocabulary=vocab,dtype=np.float32);v.fit(['placeholder']);v.idf_=packed['idf']
 qr=db.execute(f"SELECT {col} FROM read_parquet('artifacts/validation/phase4-v001/P4-SAMPLE-A.parquet') WHERE country='India' ORDER BY entity_id LIMIT 1000").fetchall();tr=db.execute(f"SELECT entity_id,{col} FROM targets_normalized WHERE country='India' LIMIT 25000").fetchall();ids=np.array([t[0] for t in tr]);q=v.transform([r[0] for r in qr]);t=v.transform([r[1] for r in tr]);start=time.perf_counter();reference=[]
 for lo in range(0,len(qr),50):
  dense=(q[lo:lo+50]@t.T).toarray()
  reference.extend(top_k(ids,row,100)[0] for row in dense)
 old=time.perf_counter()-start;start=time.perf_counter();product=sp_matmul_topn(q,t.T.tocsr(),top_n=101,threshold=0.,sort=True,n_threads=2);fast=[];ties=0
 for i in range(len(qr)):
  row=product.getrow(i);values=row.data;indices=row.indices
  if len(values)>100 and values[99]==values[100]:
   ties+=1;fast.append(top_k(ids,(q[i]@t.T).toarray().ravel(),100)[0])
  else:fast.append(top_k(ids[indices],values,100)[0])
 new=time.perf_counter()-start;different=[i for i,(left,right) in enumerate(zip(reference,fast)) if set(left)!=set(right)]
 reports.append({'field':field,'queries':len(qr),'targets':len(tr),'reference_seconds':old,'fused_seconds':new,'speedup':old/new,'different_candidate_sets':len(different),'boundary_tie_fallback_queries':ties,'target_csr_bytes':t.data.nbytes+t.indices.nbytes+t.indptr.nbytes,'scaled_10_32m_csr_gib':(t.data.nbytes+t.indices.nbytes+t.indptr.nbytes)/len(tr)*10320219/1024**3})
metadata=importlib.metadata.metadata('sparse-dot-topn');report={'routes':reports,'version':importlib.metadata.version('sparse-dot-topn'),'license':metadata.get('License-Expression') or metadata.get('License'),'source':'https://github.com/ing-bank/sparse_dot_topn','scope':'1000 India queries x25000 target records; no labels; speed is not full-pool runtime prediction; tie fallback preserves reference'};(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
