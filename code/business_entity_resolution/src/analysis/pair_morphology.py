"""Deterministic dev-only normalization and difficult-negative diagnostics.

No matcher is trained. TF-IDF negatives come from a disclosed 100k target pool;
exact-name negatives are mined against the entire training target source.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import resource
import subprocess
import time
import unicodedata
import duckdb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from src.preprocessing import representations, normalize, BASE, config_hash
LOG = logging.getLogger(__name__)


def overlap(a: set, b: set) -> tuple[float, float, float]:
    """Empty fields supply no agreement evidence."""
    intersection = len(a & b)
    return (intersection / len(a | b) if a or b else 0.,
            2 * intersection / (len(a) + len(b)) if a or b else 0.,
            intersection / min(len(a),len(b)) if a and b else 0.)


def cosine_counts(a: str, b: str, width: int) -> float:
    aa, bb = Counter(a[i:i+width] for i in range(len(a)-width+1)), Counter(b[i:i+width] for i in range(len(b)-width+1))
    denominator = math.sqrt(sum(v*v for v in aa.values()) * sum(v*v for v in bb.values()))
    return sum(v * bb.get(k,0) for k,v in aa.items()) / denominator if denominator else 0.


def features(a: str, b: str) -> dict:
    """Both missing never counts as exact matching evidence."""
    sa,sb = set(a.split()),set(b.split())
    j,d,c = overlap(sa,sb)
    result = {'jw':JaroWinkler.normalized_similarity(a,b) if a and b else 0.,
        'levenshtein':Levenshtein.normalized_similarity(a,b) if a and b else 0.,
        'token_sort':fuzz.token_sort_ratio(a,b)/100 if a and b else 0.,
        'token_set':fuzz.token_set_ratio(a,b)/100 if a and b else 0.,
        'partial':fuzz.partial_ratio(a,b)/100 if a and b else 0.,
        'jaccard':j,'dice':d,'containment':c,
        'length_ratio':min(len(a),len(b))/max(len(a),len(b)) if a or b else 0.,
        'prefix_equal':float(bool(sa and sb) and a.split()[0] == b.split()[0]),
        'suffix_equal':float(bool(sa and sb) and a.split()[-1] == b.split()[-1]),
        'acronym_equal':float(bool(sa and sb) and len(sa)>1 and ''.join(x[0] for x in a.split()) == ''.join(x[0] for x in b.split())),
        'numeric_jaccard':overlap(set(re.findall(r'\d+',a)),set(re.findall(r'\d+',b)))[0]}
    result.update({f'char{k}_cosine':cosine_counts(a,b,k) for k in range(2,6)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--queries-per-stratum',type=int,default=500);p.add_argument('--pool-size',type=int,default=100000)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'summary.json').exists(): raise FileExistsError('Use a fresh artifact directory')
    start=time.perf_counter()
    db=duckdb.connect(str(a.database),read_only=True,config={'threads':2,'memory_limit':'2GB'})
    def rows(sql):
        cursor=db.execute(sql); cols=[x[0] for x in cursor.description]
        return [dict(zip(cols,row)) for row in cursor.fetchall()]
    LOG.info('Select development queries, never fold 4')
    db.execute(f'''CREATE TEMP TABLE queries AS SELECT s.*,f.n_matches FROM train_source1 s JOIN validation_folds f ON s.entity_id=f.source1_entity_id
      WHERE fold=0 QUALIFY row_number() OVER(PARTITION BY s.country,(n_matches=0) ORDER BY sha256('morph-v1|'||entity_id))<={a.queries_per_stratum}''')
    queries=rows('SELECT * FROM queries ORDER BY entity_id'); qmap={q['entity_id']:q for q in queries}
    positives=rows('''SELECT p.source1_entity_id,p.target_id,t.business_name,t.business_address,t.country FROM positive_pairs p JOIN queries q ON q.entity_id=p.source1_entity_id JOIN train_targets t ON t.entity_id=p.target_id''')
    truth=defaultdict(set)
    for r in positives: truth[r['source1_entity_id']].add(r['target_id'])
    # A uniform hash sample is only a hard-negative discovery corpus. It is not used to claim retrieval recall.
    LOG.info('Select bounded target pool; fit text statistics only on training-owned/unowned targets')
    pool=rows(f'''SELECT t.*,o.owner_fold FROM train_targets t JOIN target_ownership o ON t.entity_id=o.target_id
      WHERE owner_fold<>4 ORDER BY sha256('morph-pool-v1|'||t.entity_id) LIMIT {a.pool_size}''')
    pairs=[]; seen=set()
    def add(qid,t,kind,label=0):
        tid=t.get('entity_id',t.get('target_id'))
        if not label and tid in truth[qid]: return
        key=(qid,tid,kind)
        if key in seen:return
        seen.add(key);q=qmap[qid]
        pairs.append({'source1_entity_id':qid,'target_id':tid,'country':q['country'],'target_country':t['country'],
          'target_source':tid[:2],'n_matches':q['n_matches'],'population':kind,'label':label,
          'name_a':q['business_name'],'name_b':t['business_name'],'address_a':q['business_address'],'address_b':t['business_address']})
    for r in positives:add(r['source1_entity_id'],r,'positive',1)
    countries={country:[t for t in pool if t['country']==country] for country in {q['country'] for q in queries}}
    for q in queries:
        h=int(hashlib.sha256(q['entity_id'].encode()).hexdigest()[:12],16)
        for k in range(3):
            add(q['entity_id'],pool[(h+k*7919)%len(pool)],'random')
            subset=countries[q['country']];add(q['entity_id'],subset[(h+k*7919)%len(subset)],'same_country')
    LOG.info('Full-target exact-name hard negatives')
    exact_negatives=rows('''SELECT q.entity_id source1_entity_id,t.entity_id,t.country,t.business_name,t.business_address
      FROM queries q JOIN targets_normalized n ON q.country=n.country AND norm(q.business_name)=n.n
      JOIN train_targets t ON t.entity_id=n.entity_id
      ANTI JOIN positive_pairs p ON p.source1_entity_id=q.entity_id AND p.target_id=t.entity_id
      WHERE n.n<>'' QUALIFY row_number() OVER(PARTITION BY q.entity_id ORDER BY jaro_winkler_similarity(norm(q.business_address),n.a) DESC,t.entity_id)<=5''')
    for t in exact_negatives:add(t['source1_entity_id'],t,'exact_name_full_pool')
    # First-token and numeric blocks are deliberately bounded to the sampled target pool.
    indexes={k:defaultdict(list) for k in ['first_token','numeric']}
    for t in pool:
        n=normalize(t['business_name'],BASE);ad=normalize(t['business_address'],BASE)
        if n:indexes['first_token'][(t['country'],n.split()[0])].append(t)
        for num in set(re.findall(r'\d+',ad)):indexes['numeric'][(t['country'],num)].append(t)
    for q in queries:
        n=normalize(q['business_name'],BASE);ad=normalize(q['business_address'],BASE)
        for kind,keys in [('first_token',[n.split()[0]] if n else []),('numeric',set(re.findall(r'\d+',ad)))]:
            candidates={t['entity_id']:t for key in keys for t in indexes[kind].get((q['country'],key),[])[:500]}
            chosen=sorted(candidates.values(),key=lambda t:(-fuzz.ratio(n,normalize(t['business_name'],BASE)),t['entity_id']))[:3]
            for t in chosen:add(q['entity_id'],t,kind+'_sample_pool')
    # Sparse products are chunked and restricted by country. Query vectors never fit IDF.
    for field in ['business_name','business_address']:
        LOG.info('TF-IDF %s hard-negative retrieval',field)
        for country,targets in countries.items():
            qs=[q for q in queries if q['country']==country]
            corpus=[normalize(t[field],BASE) for t in targets]
            vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,4),dtype=np.float32,min_df=2,max_features=150000,lowercase=False)
            vectorizer.fit([s for t,s in zip(targets,corpus) if t['owner_fold'] in [-1,1,2,3]])
            mat=vectorizer.transform(corpus).tocsr()
            for offset in range(0,len(qs),32):
                batch=qs[offset:offset+32]
                sims=(vectorizer.transform([normalize(q[field],BASE) for q in batch]) @ mat.T).tocsr()
                for i,q in enumerate(batch):
                    r=sims.getrow(i);order=np.lexsort((r.indices,-r.data))[:5]
                    for j in order:add(q['entity_id'],targets[r.indices[j]],'tfidf_'+field.split('_')[1]+'_sample_pool')
    db.close()
    pl.DataFrame(pairs).write_parquet(a.output_dir/'pairs.parquet')
    LOG.info('Rich pair features and normalization ablations: %d pairs',len(pairs))
    cache={}
    def views(s,field):
        key=(s,field)
        if key not in cache:
            result=representations(s,field);result.pop('is_null');
            result['case_only']=s.casefold();result['nfc_only']=unicodedata.normalize('NFC',s)
            cache[key]=result
        return cache[key]
    features_rows=[]; ablation=defaultdict(lambda:Counter()); examples=[]
    for i,r in enumerate(pairs):
        output={k:v for k,v in r.items() if not k.endswith(('_a','_b'))}
        for field in ['name','address']:
            va,vb=views(r[field+'_a'],field),views(r[field+'_b'],field)
            output.update({field+'_'+k:v for k,v in features(va['light'],vb['light']).items()})
            base=JaroWinkler.normalized_similarity(va['light'],vb['light']) if va['light'] and vb['light'] else 0.
            for variant in va:
                exact=bool(va[variant]) and va[variant]==vb[variant]
                jw=JaroWinkler.normalized_similarity(va[variant],vb[variant]) if va[variant] and vb[variant] else 0.
                key=(r['country'],r['target_source'],r['population'],field,variant)
                ablation[key].update({'pairs':1,'exact':int(exact),'high_jw':int(jw>=.9),'jw_improved':int(jw>base+.01),'jw_worsened':int(jw<base-.01)})
                output[field+'_'+variant+'_exact']=int(exact)
            if r['label']==0 and va['aggressive']==vb['aggressive'] and va['aggressive'] and va['light']!=vb['light'] and len(examples)<100:
                examples.append({**r,'field':field,'collision_view':va['aggressive']})
        output['name_x_address_jw']=output['name_jw']*output['address_jw']
        output['both_missing_address']=int(not r['address_a'].strip() and not r['address_b'].strip())
        output['one_missing_address']=int(not r['address_a'].strip() or not r['address_b'].strip())
        features_rows.append(output)
        if i and i%10000==0:LOG.info('Features %d/%d',i,len(pairs))
    frame=pl.DataFrame(features_rows)
    # Word TF-IDF similarity: training-only/unowned target text fits IDF; all sampled pairs transformed afterward.
    for field in ['name','address']:
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,dtype=np.float32,lowercase=False,token_pattern=r'(?u)\b\w+\b')
        vec.fit([normalize(t['business_'+field],BASE) for t in pool if t['owner_fold'] in [-1,1,2,3]])
        x=vec.transform([views(r[field+'_a'],field)['light'] for r in pairs]);y=vec.transform([views(r[field+'_b'],field)['light'] for r in pairs])
        frame=frame.with_columns(pl.Series(field+'_word_tfidf_cosine',np.asarray(x.multiply(y).sum(axis=1)).ravel()))
    frame.write_parquet(a.output_dir/'features.parquet')
    ablations=[dict(zip(['country','target_source','population','field','variant'],k),**dict(v)) for k,v in ablation.items()]
    pl.DataFrame(ablations).write_csv(a.output_dir/'normalization_ablations.csv')
    diagnostics=[]
    fnames=[c for c,d in frame.schema.items() if d.is_numeric() and c not in ['n_matches','label']]
    for population in sorted(frame['population'].unique()):
        if population=='positive':continue
        for country in sorted(frame['country'].unique()):
            data=frame.filter((pl.col('country')==country)&pl.col('population').is_in(['positive',population]))
            labels=data['label'].to_numpy()
            if len(set(labels))<2:continue
            for col in fnames:
                scores=data[col].to_numpy();pos=scores[labels==1];neg=scores[labels==0]
                diagnostics.append({'country':country,'negative_population':population,'feature':col,
                    'positives':len(pos),'negatives':len(neg),'roc_auc':roc_auc_score(labels,scores),
                    'average_precision':average_precision_score(labels,scores),'ks':ks_2samp(pos,neg).statistic,
                    'positive_mean':float(pos.mean()),'negative_mean':float(neg.mean()),
                    'fpr_at_0_9':float(np.mean(neg>=.9)),'tpr_at_0_9':float(np.mean(pos>=.9))})
    pl.DataFrame(diagnostics).write_csv(a.output_dir/'feature_diagnostics.csv')
    # Local examples aid visual diagnosis; never sent to an external service.
    pl.DataFrame(examples).write_csv(a.output_dir/'aggressive_collision_examples.tsv',separator='\t') if examples else None
    hardest=frame.filter(pl.col('label')==1).sort('name_x_address_jw').head(50)
    hardids=set(zip(hardest['source1_entity_id'],hardest['target_id']))
    pl.DataFrame([r for r in pairs if r['label'] and (r['source1_entity_id'],r['target_id']) in hardids]).write_csv(a.output_dir/'difficult_positive_examples.tsv',separator='\t')
    summary={'created_at':datetime.now(timezone.utc).isoformat(),'seed':'morph-v1 / morph-pool-v1 SHA256',
      'config_hash':config_hash({'module':'pair_morphology_v1','queries_per_stratum':a.queries_per_stratum,'pool_size':a.pool_size}),
      'scope':'Fold 0 development only; queries stratified equally by country and singleton status. Pair diagnostics conditional on sampling, not deployment prevalence. Exact-name negatives use all training targets; other blocked/TF-IDF negatives use a 100k target pool excluding fold 4. Training folds 1–3 and unowned targets only fit TF-IDF. No positive targets injected into that pool. No retrieval recall claim for sample-pool negatives. No learned matcher or threshold selection.',
      'queries':len(queries),'target_pool':len(pool),'pair_counts':frame.group_by(['country','population']).len().sort(['country','population']).to_dicts(),
      'runtime_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
      'full_pool_exact_negative_queries':len(set(x['source1_entity_id'] for x in exact_negatives)),
      'limitations':['No exhaustive all-pairs fuzzy duplicate scan','No fold-4 validation score','No supervised France labels','No production recall estimate from bounded negative pool']}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    LOG.info('Done: %s',summary['runtime_seconds'])

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');main()
