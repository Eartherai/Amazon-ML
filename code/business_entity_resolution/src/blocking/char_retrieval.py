"""Exact shard-merged character TF-IDF top-K over the full training target pool.

Vocabulary/IDF fitting uses only deterministic sampled training-owned/unowned
records. Scoring includes every target in the same country as each query.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json, time, hashlib, resource, sys
import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.token_candidates import summarize_candidates


def top_k(ids: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray,np.ndarray]:
    """Positive-score top K with lexical target-ID tie breaking, including boundary ties."""
    keep=np.flatnonzero(scores>0)
    if len(keep)>k:
        threshold=np.partition(scores[keep],len(keep)-k)[len(keep)-k]
        keep=keep[scores[keep]>=threshold]
    order=np.lexsort((ids[keep],-scores[keep]))[:k]
    return ids[keep][order],scores[keep][order]


def fused_top_k_rows(queries, targets, ids, k, threads=2, transposed=False):
    """Fused sparse top-K; reference fallback for ties/near-ties at the boundary."""
    from sparse_dot_topn import sp_matmul_topn
    transposed=targets if transposed else targets.T.tocsr()
    product=sp_matmul_topn(queries,transposed,top_n=min(k+1,len(ids)),threshold=0.,sort=True,n_threads=threads)
    results=[]
    for j in range(queries.shape[0]):
        row=product.getrow(j)
        if len(row.data)>k and abs(float(row.data[k-1])-float(row.data[k]))<=1e-6:
            results.append(top_k(ids,(queries[j]@transposed).toarray().ravel(),k))
        else:
            results.append(top_k(ids[row.indices],row.data,k))
    return results


def retrieve(con, queries, field, ngram, config, output):
    start=time.perf_counter(); col={'name':'n','address':'a'}[field]
    table=config.get('target_table','targets_normalized')
    if table not in {'targets_normalized','translit_targets'}:raise ValueError('Unsupported target view')
    fit_folds=config.get('fit_owner_folds',[-1,1,2,3])
    if not fit_folds or any(type(f) is not int or f not in [-1,0,1,2,3] for f in fit_folds):
        raise ValueError('Invalid fit folds; locked fold4 cannot fit IDF')
    fit_sql=','.join(str(f) for f in fit_folds)
    fit=con.execute(f"""SELECT t.entity_id,t.{col} FROM {table} t
        JOIN target_ownership o ON o.target_id=t.entity_id
        WHERE o.owner_fold IN ({fit_sql}) AND substr(sha256(t.entity_id),1,2) IN ('00','01','02','03')
        ORDER BY t.entity_id""").fetchall()
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(ngram,ngram),lowercase=False,
        min_df=2,max_features=config['max_features'],dtype=np.float32,norm='l2')
    vectorizer.fit([v for _,v in fit]); fit_count=len(fit)
    np.savez_compressed(output/f'{field}_char{ngram}_idf.npz',idf=vectorizer.idf_,
        vocabulary=np.array(vectorizer.get_feature_names_out()))
    fit_sha=hashlib.sha256('\n'.join(i for i,_ in fit).encode()).hexdigest(); del fit
    k=config['top_k_per_route']; results=[]; rows_scored=0
    for country in sorted({q['country'] for q in queries}):
        qs=[q for q in queries if q['country']==country]
        matrix=vectorizer.transform([q[field] for q in qs])
        best=[(np.array([],dtype='U32'),np.array([],dtype=np.float32)) for q in qs]
        cursor=con;cursor.execute(f'SELECT entity_id,{col} FROM {table} WHERE country=?',[country])
        while batch:=cursor.fetchmany(config['chunk_rows']):
            if time.perf_counter()-start>config['route_runtime_cap_seconds']:
                raise TimeoutError('Route runtime cap reached; partial result is not a full-pool route')
            ids=np.array([r[0] for r in batch]); targets=vectorizer.transform([r[1] for r in batch])
            fused_targets=targets.T.tocsr() if config.get('kernel','reference')=='fused' else None
            # Query blocks bound the dense working product; the full sparse index is never retained.
            for lo in range(0,len(qs),config['query_block_rows']):
                block=matrix[lo:lo+config['query_block_rows']]
                if config.get('kernel','reference')=='fused':
                    selected=fused_top_k_rows(block,fused_targets,ids,k,config.get('kernel_threads',2),transposed=True)
                else:
                    selected=[top_k(ids,values,k) for values in (block@targets.T).toarray()]
                for j,(new_ids,new_scores) in enumerate(selected):
                    ix=lo+j
                    best[ix]=top_k(np.concatenate((best[ix][0],new_ids)),np.concatenate((best[ix][1],new_scores)),k)
            rows_scored+=len(batch)
        for q,(ids,scores) in zip(qs,best):
            results.extend((q['entity_id'],str(i),f'{field}_char{ngram}',float(s),rank+1,'light_nfc',str(i)[:2])
                           for rank,(i,s) in enumerate(zip(ids,scores)))
        print(json.dumps({'route':f'{field}_char{ngram}','country':country,'targets_scored':rows_scored,
            'seconds':time.perf_counter()-start}),flush=True)
    return results,{'field':field,'ngram':ngram,'fit_rows':fit_count,'fit_ids_sha256':fit_sha,
        'fit_owner_folds':fit_folds,'vocabulary_size':len(vectorizer.vocabulary_),'target_rows_scored':rows_scored,
        'seconds':time.perf_counter()-start}


def run(database,config_path,output):
    cfg=json.loads(config_path.read_text());output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();con=duckdb.connect(str(database),read_only=True,config={'threads':2,'memory_limit':'1GB'})
    qrows=con.execute('SELECT entity_id,country,n,a FROM read_parquet(?) ORDER BY entity_id',[cfg['queries_path']]).fetchall()
    queries=[dict(zip(('entity_id','country','name','address'),r)) for r in qrows]
    con.execute('CREATE TEMP TABLE queries AS SELECT * FROM read_parquet(?)',[cfg['queries_path']])
    truth={q['entity_id']:set() for q in queries}
    for q,t in con.execute('SELECT p.source1_entity_id,p.target_id FROM positive_pairs p JOIN queries q ON q.entity_id=p.source1_entity_id').fetchall():truth[q].add(t)
    pool=con.execute('SELECT count(*) FROM targets_normalized').fetchone()[0]
    token={q['entity_id']:set() for q in queries}
    for q,t in con.execute('SELECT source1_entity_id,target_id FROM read_parquet(?)',[cfg['token_candidates_path']]).fetchall():token[q].add(t)
    union={q:set(ts) for q,ts in token.items()}; char_union={q:set() for q in truth}
    report={'config':cfg,'pool_count':pool,'routes':{},'metrics':{},'limitations':[
        '1000 balanced-country development queries, not locked evaluation or population-weighted score',
        'Vocabulary/IDF fitted on deterministic approximately 1/64 training-eligible target sample, not all training text',
        'All 10320219 targets transformed; same-country retrieval only; no learned matcher',
        'TopK is per route across both target sources; zero-cosine pairs omitted',
        'Oracle scores are upper bounds, not model scores; floating point float32 cosine']}
    def metric(candidates):
        m=summarize_candidates(queries,truth,candidates,pool)
        m['candidate_quantiles']['p90']=float(np.quantile([len(candidates[q]) for q in truth],.9))
        return m
    for field,ngram in cfg['routes']:
        route=f'{field}_char{ngram}'
        try:rows,metadata=retrieve(con,queries,field,ngram,cfg,output)
        except TimeoutError as exc:
            report['routes'][route]={'status':'timeout','error':str(exc)};break
        con.execute('CREATE OR REPLACE TEMP TABLE route_rows(source1_entity_id VARCHAR,target_id VARCHAR,route VARCHAR,route_score FLOAT,route_rank INTEGER,representation VARCHAR,target_source VARCHAR)')
        con.executemany('INSERT INTO route_rows VALUES (?,?,?,?,?,?,?)',rows)
        con.execute('COPY route_rows TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/f'{route}.parquet')])
        candidates={q:set() for q in truth}
        for q,t,*_ in rows:candidates[q].add(t);union[q].add(t);char_union[q].add(t)
        metadata['metrics']=metric(candidates);report['routes'][route]=metadata
        report['metrics']['char_union']=metric(char_union);report['metrics']['token_char_union']=metric(union)
        for cap in [20,50,100]:
            capped={q:set() for q in truth}
            for q,t,_,_,rank,*_ in rows:
                if rank<=cap:capped[q].add(t)
            metadata.setdefault('by_k',{})[str(cap)]=metric(capped)
        report['runtime_seconds']=time.perf_counter()-start
        report['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**(3 if sys.platform=='darwin' else 2))
        (output/'metrics.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'completed':route,'recall':metadata['metrics']['link_recall'],'union':report['metrics']['token_char_union']['link_recall']}),flush=True)
    con.execute('CREATE TEMP TABLE union_rows(source1_entity_id VARCHAR,target_id VARCHAR)')
    con.executemany('INSERT INTO union_rows VALUES (?,?)',[(q,t) for q in sorted(union) for t in sorted(union[q])])
    con.execute('COPY union_rows TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'token_char_union.parquet')])
    (output/'metrics.json').write_text(json.dumps(report,indent=2));con.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--database',type=Path,default=Path('artifacts/audit.duckdb'));p.add_argument('--config',type=Path,default=Path('configs/blocking/CHAR-001.json'));p.add_argument('--output',type=Path,default=Path('outputs/candidates/CHAR-001'));a=p.parse_args();run(a.database,a.config,a.output)
