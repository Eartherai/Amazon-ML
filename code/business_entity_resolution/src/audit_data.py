"""Full-file audit with bounded DuckDB memory; no training or external record lookup."""
import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path
import duckdb

LOG = logging.getLogger(__name__)
COLS = ['entity_id','business_name','business_address','country']

def sqlstr(value):
    return "'" + str(value).replace("'", "''") + "'"

def rows(con, sql):
    cur=con.execute(sql)
    names=[d[0] for d in cur.description]
    return [dict(zip(names,r)) for r in cur.fetchall()]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()

def connect(db):
    con=duckdb.connect(str(db))
    con.execute("SET threads=4; SET memory_limit='6GB'; SET max_temp_directory_size='6GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE MACRO norm(x) AS trim(regexp_replace(lower(nfc_normalize(coalesce(x,''))), '[^\\p{L}\\p{M}\\p{N}]+', ' ', 'g'))")
    return con

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); a.database.parent.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'audit.json').exists(): raise FileExistsError('Use a new output directory for a new audit')
    start=time.perf_counter(); con=connect(a.database); report={'files':{},'notes':['All per-file counts exact, lengths exact quantiles. Strings preserved; blank TSV cells are empty strings, no implicit NA conversion. Postal tokens are shape proxies, not geocoded postal codes.']}
    for split in ('train','test'):
        for source in (1,2,3):
            name=f'{split}_source{source}'; path=a.data_dir/split/(name+'.tsv')
            LOG.info('Loading %s',name)
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM read_csv({sqlstr(path)}, delim='\t', header=true, all_varchar=true, nullstr='__AUDIT_IMPOSSIBLE_NULL_SENTINEL__', strict_mode=true)")
            if [r[0] for r in con.execute(f'DESCRIBE {name}').fetchall()]!=COLS: raise ValueError(f'Unexpected columns in {name}')
            stats=rows(con,f'''SELECT count(*) n, count(DISTINCT entity_id) unique_ids,
                count(DISTINCT business_name) unique_names, count(DISTINCT business_address) unique_addresses,
                sum(length(entity_id)+length(business_name)+length(business_address)+length(country)) text_characters,
                count(*) FILTER (WHERE NOT starts_with(entity_id,'S{source}-')) invalid_prefixes FROM {name}''')[0]
            stats['file_bytes']=path.stat().st_size; stats['sha256']=sha256(path)
            stats['schema']={c:'VARCHAR' for c in COLS}; stats['fields']={}
            for c in COLS:
                stats['fields'][c]=rows(con,f'''SELECT count(*) FILTER (WHERE {c} IS NULL) null_count,
                count(*) FILTER (WHERE {c}='') empty_strings,
                count(*) FILTER (WHERE trim({c})='') blank_after_trim,
                count(*) FILTER (WHERE lower(trim({c})) IN ('na','n/a','null','none','nan')) textual_missing_markers,
                avg(length({c})) mean_length, min(length({c})) min_length,max(length({c})) max_length,
                quantile_cont(length({c}),[0.01,0.25,0.5,0.75,0.95,0.99]) length_quantiles FROM {name}''')[0]
            stats['countries']=rows(con,f'SELECT country,count(*) n FROM {name} GROUP BY country ORDER BY n DESC')
            stats['exact_duplicates']=rows(con,f'''SELECT count(*) duplicate_groups, coalesce(sum(n-1),0) redundant_rows,coalesce(max(n),0) largest_group FROM (SELECT count(*) n FROM {name} GROUP BY business_name,business_address,country HAVING count(*)>1)''')[0]
            stats['normalized_duplicates']=rows(con,f'''SELECT count(*) duplicate_groups,coalesce(sum(n-1),0) redundant_rows,coalesce(max(n),0) largest_group FROM (SELECT count(*) n FROM {name} GROUP BY norm(business_name),norm(business_address),country HAVING count(*)>1)''')[0]
            stats['text_patterns']=rows(con,f'''SELECT
            count(*) FILTER (WHERE regexp_matches(business_name,'[^\\x00-\\x7F]')) nonascii_name,
            count(*) FILTER (WHERE regexp_matches(business_address,'[^\\x00-\\x7F]')) nonascii_address,
            count(*) FILTER (WHERE regexp_matches(business_name||business_address,'[\\x{{0900}}-\\x{{097F}}]')) devanagari,
            count(*) FILTER (WHERE regexp_matches(business_name||business_address,'[\\x{{0B80}}-\\x{{0BFF}}]')) tamil,
            count(*) FILTER (WHERE regexp_matches(business_name||business_address,'[\\x{{0400}}-\\x{{04FF}}]')) cyrillic,
            count(*) FILTER (WHERE regexp_matches(business_name||business_address,'[\\x{{4E00}}-\\x{{9FFF}}]')) cjk,
            count(*) FILTER (WHERE regexp_matches(business_name||business_address,'[\\x{{0600}}-\\x{{06FF}}]')) arabic,
            count(*) FILTER (WHERE regexp_matches(business_address,'\\b[0-9]{{5}}(-[0-9]{{4}})?\\b')) postal5_like,
            count(*) FILTER (WHERE regexp_matches(business_address,'\\b[0-9]{{6}}\\b')) postal6_like,
            count(*) FILTER (WHERE regexp_matches(business_name,'[0-9]')) name_with_digits,
            count(*) FILTER (WHERE regexp_matches(business_address,'[0-9]')) address_with_digits
            FROM {name}''')[0]
            stats['numeric_token_counts']=rows(con,f"SELECT len(regexp_extract_all(business_address,'[0-9]+')) numeric_tokens,count(*) n FROM {name} GROUP BY 1 ORDER BY 1")
            stats['legal_suffixes']=rows(con,f"SELECT regexp_extract(norm(business_name),'(pvt ltd|private limited|limited|ltd|llc|inc|incorporated|corp|corporation|llp|sarl|sas|sa|gmbh)$') suffix,count(*) n FROM {name} GROUP BY 1 ORDER BY n DESC")
            report['files'][name]=stats
            (a.output_dir/'progress.json').write_text(json.dumps(report,indent=2))
            LOG.info('%s n=%s country=%s',name,stats['n'],stats['countries'])
            con.execute('CHECKPOINT')
    gt=a.data_dir/'train/train_ground_truth.tsv'
    con.execute(f"CREATE TABLE IF NOT EXISTS ground_truth AS SELECT * FROM read_csv({sqlstr(gt)},delim='\t',header=true,all_varchar=true,nullstr='__AUDIT_IMPOSSIBLE_NULL_SENTINEL__',strict_mode=true)")
    con.execute("CREATE TABLE IF NOT EXISTS positive_pairs AS SELECT source1_entity_id,unnest(string_split(matched_entity_ids,',')) target_id FROM ground_truth WHERE matched_entity_ids<>''")
    con.execute("CREATE OR REPLACE VIEW train_targets AS SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3")
    con.execute("CREATE TABLE IF NOT EXISTS label_counts AS SELECT source1_entity_id,CASE WHEN matched_entity_ids='' THEN 0 ELSE len(string_split(matched_entity_ids,',')) END n_matches,contains(matched_entity_ids,'S2-') has_s2,contains(matched_entity_ids,'S3-') has_s3 FROM ground_truth")
    report['ground_truth']={'file_bytes':gt.stat().st_size,'sha256':sha256(gt)}
    g=report['ground_truth']; g['integrity']=rows(con,"""SELECT (SELECT count(*) FROM ground_truth) record_count,
    (SELECT count(*)-count(DISTINCT source1_entity_id) FROM ground_truth) duplicate_s1,
    (SELECT count(*) FROM train_source1 ANTI JOIN ground_truth ON entity_id=source1_entity_id) missing_s1,
    (SELECT count(*) FROM ground_truth ANTI JOIN train_source1 ON entity_id=source1_entity_id) unknown_s1,
    (SELECT count(*) FROM positive_pairs) positive_pairs,
    (SELECT count(*) FROM positive_pairs ANTI JOIN train_targets ON entity_id=target_id) unknown_targets,
    (SELECT count(*) FROM (SELECT source1_entity_id,target_id FROM positive_pairs GROUP BY ALL HAVING count(*)>1)) duplicate_pairs,
    (SELECT count(*) FROM (SELECT target_id FROM positive_pairs GROUP BY 1 HAVING count(DISTINCT source1_entity_id)>1)) shared_targets""")[0]
    g['match_counts']=rows(con,'SELECT n_matches,count(*) n FROM label_counts GROUP BY 1 ORDER BY 1')
    g['source_distribution']=rows(con,'SELECT has_s2,has_s3,count(*) n FROM label_counts GROUP BY ALL ORDER BY ALL')
    g['by_country']=rows(con,"""SELECT country,count(*) n,avg(n_matches) mean_matches,count(*) FILTER (WHERE n_matches=0) singletons,count(*) FILTER (WHERE n_matches=1) one_match,count(*) FILTER (WHERE n_matches=2) two_matches,count(*) FILTER (WHERE n_matches>=3) three_plus FROM label_counts JOIN train_source1 ON source1_entity_id=entity_id GROUP BY country""")
    g['cross_country_pairs']=rows(con,"""SELECT a.country source_country,b.country target_country,count(*) n FROM positive_pairs p JOIN train_source1 a ON a.entity_id=p.source1_entity_id JOIN train_targets b ON b.entity_id=p.target_id GROUP BY ALL""")
    report['train_test_overlap']={}
    for s in (1,2,3):
        report['train_test_overlap'][f'source{s}']=rows(con,f'''SELECT
        (SELECT count(*) FROM test_source{s} t SEMI JOIN train_source{s} r USING(entity_id)) shared_ids,
        (SELECT count(*) FROM test_source{s} t SEMI JOIN train_source{s} r USING(business_name,business_address,country)) exact_record_test_rows''')[0]
    g['ownership_counts']=rows(con,'SELECT owners,count(*) targets FROM (SELECT target_id,count(DISTINCT source1_entity_id) owners FROM positive_pairs GROUP BY 1) GROUP BY 1 ORDER BY 1')
    report['elapsed_seconds']=time.perf_counter()-start
    report['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)
    report['database_bytes']=a.database.stat().st_size
    (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2))
    con.execute('CHECKPOINT'); con.close(); LOG.info('Audit complete %.1fs',report['elapsed_seconds'])
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s'); main()
