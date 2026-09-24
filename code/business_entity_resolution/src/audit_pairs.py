"""Full positive-pair diagnostics plus explicitly bounded near-duplicate probes."""
import argparse,json,logging,time
from pathlib import Path
from src.audit_data import connect,rows,sqlstr
LOG=logging.getLogger(__name__)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'pairs.json').exists():raise FileExistsError('Use a fresh output path')
    t=time.perf_counter();c=connect(a.database)
    for name,base in [('s1_normalized','train_source1'),('targets_normalized','train_targets')]:
        LOG.info('Normalize %s',base)
        c.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT entity_id,country,norm(business_name) n,norm(business_address) a FROM {base}")
    c.execute("CREATE OR REPLACE MACRO nums(x) AS list_distinct(regexp_extract_all(x,'[0-9]+'))")
    c.execute("CREATE OR REPLACE MACRO postal(x) AS list_distinct(regexp_extract_all(x,'\\b[0-9]{5,6}\\b'))")
    c.execute("CREATE OR REPLACE MACRO acronym(x) AS array_to_string(list_transform(string_split(x,' '), t -> left(t,1)),'')")
    LOG.info('Measure all positive pairs')
    c.execute('''CREATE TABLE positive_features AS SELECT p.source1_entity_id,p.target_id,s.country,
    s.n=t.n AND s.n<>'' exact_name,s.a=t.a AND s.a<>'' exact_address,
    jaro_winkler_similarity(s.n,t.n) name_jw,jaro_winkler_similarity(s.a,t.a) address_jw,
    len(list_intersect(nums(s.a),nums(t.a)))>0 numeric_overlap,
    len(nums(s.a))>0 AND len(nums(t.a))>0 numeric_both_present,
    len(list_intersect(postal(s.a),postal(t.a)))>0 postal_overlap,
    len(postal(s.a))>0 AND len(postal(t.a))>0 postal_both_present,
    acronym(s.n)=acronym(t.n) AND length(acronym(s.n))>1 acronym_equal,
    strip_accents(s.n)=strip_accents(t.n) AND s.n<>'' accent_fold_equal,
    regexp_matches(t.n,'[^\\x00-\\x7F]') nonascii_target_name,
    s.a='' OR t.a='' missing_address,
    len(list_intersect(string_split(s.n,' '),string_split(t.n,' ')))::DOUBLE/nullif(len(list_distinct(list_concat(string_split(s.n,' '),string_split(t.n,' ')))),0) name_jaccard
    FROM positive_pairs p JOIN s1_normalized s ON s.entity_id=p.source1_entity_id JOIN targets_normalized t ON t.entity_id=p.target_id''')
    agg='''count(*) pairs,avg(exact_name::int) exact_name_rate,avg(exact_address::int) exact_address_rate,
    avg((name_jw>=0.9)::int) high_name_jw_rate,avg((address_jw>=0.9)::int) high_address_jw_rate,
    avg(numeric_overlap::int) numeric_overlap_rate,avg(numeric_both_present::int) numeric_both_present_rate,
    avg(postal_overlap::int) postal_overlap_rate,avg(postal_both_present::int) postal_both_present_rate,
    avg(acronym_equal::int) acronym_equal_rate,avg(accent_fold_equal::int) accent_fold_equal_rate,
    avg(nonascii_target_name::int) nonascii_target_name_rate,avg(missing_address::int) missing_address_rate,
    avg((name_jw<0.7 AND address_jw<0.7)::int) both_jw_under_07_rate,
    avg((name_jw<0.7)::int) name_jw_under_07_rate,
    quantile_cont(name_jw,[0.01,0.1,0.5,0.9,0.99]) name_jw_quantiles,
    quantile_cont(address_jw,[0.01,0.1,0.5,0.9,0.99]) address_jw_quantiles'''
    report={'positive_all':rows(c,'SELECT '+agg+' FROM positive_features')[0],
            'positive_by_country':rows(c,'SELECT country,'+agg+' FROM positive_features GROUP BY country'),
            'positive_by_target_source':rows(c,"SELECT left(target_id,2) target_source,"+agg+' FROM positive_features GROUP BY 1')}
    LOG.info('Inspect exact name block volume and hard negatives')
    report['exact_name_block_volume']=rows(c,"""SELECT sum(s.cnt*t.cnt) pairs,max(s.cnt*t.cnt) largest_block_pairs FROM
    (SELECT country,n,count(*) cnt FROM s1_normalized WHERE n<>'' GROUP BY ALL) s
    JOIN (SELECT country,n,count(*) cnt FROM targets_normalized WHERE n<>'' GROUP BY ALL) t USING(country,n)""")[0]
    c.execute('''CREATE TEMP TABLE probe_s1 AS SELECT * FROM s1_normalized WHERE substr(sha256(entity_id),1,3) IN ('000','001','002','003','004','005','006','007')''')
    c.execute('''CREATE TABLE hard_negative_probe AS SELECT s.entity_id source1_entity_id,t.entity_id target_id,s.country,
    jaro_winkler_similarity(s.a,t.a) address_jw,s.a=t.a AND s.a<>'' exact_address
    FROM probe_s1 s JOIN targets_normalized t USING(country,n)
    ANTI JOIN positive_pairs p ON p.source1_entity_id=s.entity_id AND p.target_id=t.entity_id WHERE s.n<>'' ''')
    report['hard_negative_probe']={'scope':'SHA256 prefix 000..007 S1 sample against ALL training S2/S3, exact normalized name plus country, excluding labeled positives. Not prevalence across all negatives.',
        'sample_s1':c.execute('SELECT count(*) FROM probe_s1').fetchone()[0],
        'statistics':rows(c,'SELECT country,count(*) pairs,count(DISTINCT source1_entity_id) affected_s1,avg(exact_address::int) exact_address_rate,avg((address_jw>=0.9)::int) high_address_jw_rate FROM hard_negative_probe GROUP BY country')}
    report['near_duplicate_probes']={}
    for split in ['train','test']:
        for source in [1,2,3]:
            name=f'{split}_source{source}';LOG.info('Near-neighbor diagnostic %s',name)
            report['near_duplicate_probes'][name]=rows(c,f'''WITH normalized AS (SELECT entity_id,country,norm(business_name) n,norm(business_address) a FROM {name}),
            adjacent AS (SELECT *,lag(n) OVER w prev_n,lag(a) OVER w prev_a FROM normalized WINDOW w AS (PARTITION BY country ORDER BY n,a,entity_id))
            SELECT count(*) FILTER (WHERE prev_n IS NOT NULL) compared_pairs,
            count(*) FILTER (WHERE n<>'' AND a<>'' AND prev_n<>'' AND prev_a<>'' AND (n<>prev_n OR a<>prev_a) AND jaro_winkler_similarity(n,prev_n)>=0.9 AND jaro_winkler_similarity(a,prev_a)>=0.9) near_pairs FROM adjacent''')[0]
    report['near_duplicate_scope']='Exact count for adjacent records sorted by country, normalized name, normalized address, ID within each source; JW>=0.90 both fields, excluding identical normalized pairs and blank fields. Lower-bound diagnostic, not exhaustive fuzzy clustering.'
    report['elapsed_seconds']=time.perf_counter()-t
    for title,query in {
        'difficult_positive_examples':'''SELECT f.*,s.business_name source_name,t.business_name target_name,s.business_address source_address,t.business_address target_address FROM positive_features f JOIN train_source1 s ON s.entity_id=f.source1_entity_id JOIN train_targets t ON t.entity_id=f.target_id WHERE name_jw<0.7 AND address_jw<0.7 ORDER BY sha256(source1_entity_id||target_id) LIMIT 60''',
        'deceptive_negative_examples':'''SELECT f.*,s.business_name source_name,t.business_name target_name,s.business_address source_address,t.business_address target_address FROM hard_negative_probe f JOIN train_source1 s ON s.entity_id=f.source1_entity_id JOIN train_targets t ON t.entity_id=f.target_id ORDER BY address_jw DESC,source1_entity_id,target_id LIMIT 60'''
    }.items():
        c.execute(f"COPY ({query}) TO {sqlstr(a.output_dir/(title+'.tsv'))} (HEADER,DELIMITER '\t')")
    (a.output_dir/'pairs.json').write_text(json.dumps(report,indent=2));c.execute('CHECKPOINT');LOG.info('Done %.1fs',report['elapsed_seconds'])
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');main()
