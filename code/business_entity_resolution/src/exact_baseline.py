"""No-training exact name/address diagnostic on development entities only."""
import argparse,json,resource,subprocess,time
from pathlib import Path
from src.audit_data import connect,rows,sqlstr
from src.evaluation import f05_from_counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.output_dir.mkdir(parents=True,exist_ok=False)
    if cfg['fold']!=0:raise ValueError('Diagnostic baseline is restricted to development fold 0')
    start=time.perf_counter();c=connect(a.database)
    c.execute('''CREATE TEMP TABLE exact_candidates AS SELECT s.entity_id source1_entity_id,t.entity_id target_id
    FROM s1_normalized s JOIN validation_folds f ON s.entity_id=f.source1_entity_id AND f.fold=0
    JOIN targets_normalized t ON s.country=t.country AND s.n=t.n AND s.a=t.a
    WHERE s.n<>'' AND s.a<>'' ''')
    c.execute('''CREATE TEMP TABLE predictions_summary AS WITH pred AS
    (SELECT q.source1_entity_id,count(*) predicted,count(p.target_id) tp FROM exact_candidates q LEFT JOIN positive_pairs p USING(source1_entity_id,target_id) GROUP BY 1)
    SELECT f.source1_entity_id,f.country,f.n_matches actual,coalesce(p.predicted,0) predicted,coalesce(p.tp,0) tp,
    CASE WHEN f.n_matches=0 AND coalesce(p.predicted,0)=0 THEN 1.0 ELSE 1.25*coalesce(p.tp,0)/(coalesce(p.predicted,0)+0.25*f.n_matches) END f0_5
    FROM validation_folds f LEFT JOIN pred p USING(source1_entity_id) WHERE f.fold=0''')
    agg='''count(*) entities,avg(f0_5) macro_f0_5,sum(tp)::DOUBLE/nullif(sum(predicted),0) micro_precision,
    sum(tp)::DOUBLE/nullif(sum(actual),0) micro_recall,
    avg(f0_5) FILTER(WHERE actual=0) singleton_f0_5,
    avg(f0_5) FILTER(WHERE actual>0) non_singleton_f0_5,
    sum(predicted) candidate_pairs,sum(tp) tp,sum(actual) true_links,
    avg(predicted) candidates_mean,quantile_cont(predicted,[0.5,0.95,0.99]) candidates_p50_p95_p99,
    avg(CASE WHEN actual=0 THEN 1 ELSE 1.25*tp/(tp+0.25*actual) END) candidate_oracle_macro_f0_5'''
    report={'config':cfg,'overall':rows(c,'SELECT '+agg+' FROM predictions_summary')[0],
        'by_country':rows(c,'SELECT country,'+agg+' FROM predictions_summary GROUP BY country'),
        'all_empty_control':rows(c,'SELECT avg((actual=0)::int) macro_f0_5 FROM predictions_summary')[0],
        'candidate_pool_targets':c.execute('SELECT count(*) FROM targets_normalized').fetchone()[0],
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    v=report['overall'];v['candidate_recall']=v['micro_recall'];v['reduction_ratio']=1-v['candidate_pairs']/(v['entities']*report['candidate_pool_targets'])
    # Independently verify count-form scoring on a deterministic sample including singletons.
    check=c.execute('SELECT tp,predicted,actual,f0_5 FROM predictions_summary ORDER BY sha256(source1_entity_id) LIMIT 10000').fetchall()
    assert all(abs(f05_from_counts(tp,pred,actual)-score)<1e-12 for tp,pred,actual,score in check)
    c.execute(f"COPY (SELECT * FROM exact_candidates ORDER BY source1_entity_id,target_id) TO {sqlstr(a.output_dir/'scored_candidates.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    c.execute(f"COPY (SELECT * FROM predictions_summary ORDER BY source1_entity_id) TO {sqlstr(a.output_dir/'entity_scores.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    report['runtime_seconds']=time.perf_counter()-start
    report['peak_ram_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)
    (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
