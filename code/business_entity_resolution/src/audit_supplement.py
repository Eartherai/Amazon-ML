"""Script coverage and a deterministic generic transliteration audit sample."""
import argparse,json,subprocess,time
from pathlib import Path
from rapidfuzz.distance import JaroWinkler
from src.audit_data import connect,rows
from src.normalization import light

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--transliterator',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=connect(a.database);report={'script_coverage':{}}
    scripts={'devanagari':('0900','097F'),'bengali':('0980','09FF'),'gurmukhi':('0A00','0A7F'),'gujarati':('0A80','0AFF'),'oriya':('0B00','0B7F'),'tamil':('0B80','0BFF'),'telugu':('0C00','0C7F'),'kannada':('0C80','0CFF'),'malayalam':('0D00','0D7F')}
    terms=[f"count(*) FILTER(WHERE regexp_matches(business_name||business_address,'[\\x{{{lo}}}-\\x{{{hi}}}]')) {name}" for name,(lo,hi) in scripts.items()]
    for split in ['train','test']:
        for source in [1,2,3]:
            name=f'{split}_source{source}'
            report['script_coverage'][name]=rows(c,'SELECT '+','.join(terms)+' FROM '+name)[0]
    sample=c.execute('''SELECT p.source1_entity_id,p.target_id,s.business_name,t.business_name,s.country
    FROM positive_pairs p JOIN train_source1 s ON s.entity_id=p.source1_entity_id
    JOIN train_targets t ON t.entity_id=p.target_id
    WHERE regexp_matches(t.business_name,'[^\\x00-\\x7F]')
    ORDER BY sha256(p.source1_entity_id||'|'||p.target_id) LIMIT 10000''').fetchall()
    payload=''.join(json.dumps([s[2],s[3]],ensure_ascii=False)+'\n' for s in sample)
    start=time.perf_counter()
    proc=subprocess.run([str(a.transliterator.resolve())],input=payload,text=True,capture_output=True,check=True,timeout=120)
    converted=[json.loads(line) for line in proc.stdout.splitlines()]
    assert len(converted)==len(sample)
    cases=[]
    for (sid,tid,s,t,country),(ss,tt) in zip(sample,converted):
        original=JaroWinkler.normalized_similarity(light(s),light(t))
        changed=JaroWinkler.normalized_similarity(light(ss),light(tt))
        cases.append({'country':country,'original':original,'transliterated':changed,'exact':light(ss)==light(tt)})
    by={}
    for country in sorted({x['country'] for x in cases}):
        r=[x for x in cases if x['country']==country];n=len(r)
        by[country]={'pairs':n,'original_jw_ge_09':sum(x['original']>=.9 for x in r)/n,'transliterated_jw_ge_09':sum(x['transliterated']>=.9 for x in r)/n,
            'original_low_transliterated_high':sum(x['original']<.7 and x['transliterated']>=.9 for x in r)/n,'transliterated_exact':sum(x['exact'] for x in r)/n}
    report['transliteration_probe']={'scope':'First 10000 labeled positive pairs sorted by SHA256(S1|target) among non-ASCII target business names. Conditional sample, NOT an overall rate or candidate recall. No negatives tested in this probe.',
      'engine':'macOS Foundation StringTransform Any-Latin; Latin-ASCII (ICU), runtime macOS 27.0; Linux ICU parity is unverified',
      'n':len(cases),'runtime_seconds':time.perf_counter()-start,'by_country':by}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report['transliteration_probe'],indent=2))
if __name__=='__main__':main()
