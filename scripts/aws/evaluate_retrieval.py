"""Exact streaming SQL retrieval evaluation, explicitly excluding locked Fold4 labels."""
import argparse,json,time
from pathlib import Path
import duckdb


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    con=duckdb.connect(str(a.output/'work.duckdb'),config={'threads':8,'memory_limit':'16GB'})
    con.read_parquet(str(a.candidates/'query_coverage.parquet')).create_view('coverage')
    con.read_parquet(str(a.labels/'queries.parquet')).create_view('queries')
    con.read_parquet(str(a.labels/'truth.parquet')).create_view('truth')
    if con.execute('SELECT count(*) FROM queries WHERE fold NOT IN (0,1,2,3) OR fold IS NULL').fetchone()[0]:raise ValueError('Locked/invalid labels supplied')
    if con.execute('SELECT count(*) FROM queries q ANTI JOIN coverage c ON q.source1_entity_id=c.entity_id').fetchone()[0]:raise ValueError('Missing evaluation queries')
    for table,key in [('coverage','entity_id'),('queries','source1_entity_id')]:
        if con.execute(f'SELECT count(*)-count(DISTINCT {key}) FROM {table}').fetchone()[0]:raise ValueError('Duplicate query IDs')
    con.read_parquet(str(a.candidates/'*-c*-s*-b*.parquet')).create_view('routes')
    con.execute('CREATE TABLE candidates AS SELECT DISTINCT source1_entity_id,target_id FROM routes')
    if con.execute('SELECT count(*) FROM truth t ANTI JOIN queries q USING(source1_entity_id)').fetchone()[0]:raise ValueError('Truth outside unlocked queries')
    if con.execute('SELECT count(*) FROM candidates c ANTI JOIN coverage q ON c.source1_entity_id=q.entity_id').fetchone()[0]:raise ValueError('Unknown candidate query')
    con.execute('CREATE TABLE hits AS SELECT t.source1_entity_id,count(*) AS hit FROM truth t JOIN candidates c USING(source1_entity_id,target_id) GROUP BY t.source1_entity_id')
    con.execute('CREATE TABLE counts AS SELECT source1_entity_id,count(*) AS count FROM candidates GROUP BY source1_entity_id')
    con.execute('CREATE TABLE scores AS SELECT q.*,coalesce(h.hit,0) AS hits,coalesce(c.count,0) AS candidates,CASE WHEN q.n_matches=0 THEN 1.0 ELSE 1.25*coalesce(h.hit,0)/(coalesce(h.hit,0)+0.25*q.n_matches) END AS oracle FROM queries q LEFT JOIN hits h USING(source1_entity_id) LEFT JOIN counts c USING(source1_entity_id)')
    def summary(where='TRUE'):
        cur=con.execute(f'''SELECT count(*) AS queries,sum(n_matches) AS true_links,sum(hits) AS retrieved_links,sum(hits)::DOUBLE/nullif(sum(n_matches),0) AS link_recall,avg(CASE WHEN n_matches>0 THEN (hits=n_matches)::INT END) AS complete_entity_recall,avg(CASE WHEN n_matches>0 THEN (hits>0)::INT END) AS any_link_entity_recall,avg(oracle) AS oracle_macro_f0_5,count(*) FILTER(WHERE n_matches=0) AS singleton_count FROM scores WHERE {where}''')
        return dict(zip([x[0] for x in cur.description],cur.fetchone()))
    report={'scope':'All supplied S1 candidate generation; label-based metrics ONLY folds0–3. Fold4 remains CLOSED.','evaluation':summary(),'by_country':{},'by_match_count':{}}
    for country, in con.execute('SELECT DISTINCT country FROM scores ORDER BY country').fetchall():report['by_country'][country]=summary("country='"+country.replace("'","''")+"'")
    for name,condition in [('0','n_matches=0'),('1','n_matches=1'),('2','n_matches=2'),('3-5','n_matches BETWEEN 3 AND 5'),('6-10','n_matches BETWEEN 6 AND 10'),('11+','n_matches>10')]:report['by_match_count'][name]=summary(condition)
    cur=con.execute('SELECT count(*) AS full_s1_count,sum(coalesce(c.count,0)) AS total_candidate_pairs,avg(coalesce(c.count,0)) AS mean_candidates,quantile_cont(coalesce(c.count,0),[0.5,0.9,0.95,0.99,0.999]) AS candidate_quantiles,max(coalesce(c.count,0)) AS max_candidates FROM coverage q LEFT JOIN counts c ON q.entity_id=c.source1_entity_id')
    report['unlabeled_full_coverage']=dict(zip([x[0] for x in cur.description],cur.fetchone()));report['seconds']=time.perf_counter()-start
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    con.close()
if __name__=='__main__':main()
