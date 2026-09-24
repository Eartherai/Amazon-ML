"""Apply a local ICU-backed binary to frozen development pair samples."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import subprocess
import time
import unicodedata
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from src.preprocessing import normalize, BASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    a=p.parse_args();dest=a.pairs/'transliteration.json'
    if dest.exists():raise FileExistsError(dest)
    start=time.perf_counter();frame=pl.read_parquet(a.pairs/'pairs.parquet');raw=frame.to_dicts()
    texts=sorted({r[k] for r in raw for k in ['name_a','name_b','address_a','address_b']})
    # Existing Swift script reads one JSON array per line, transforms each string independently.
    output=subprocess.run([str(a.binary.resolve())],input='\n'.join(json.dumps(texts[i:i+1000],ensure_ascii=False) for i in range(0,len(texts),1000))+'\n',text=True,capture_output=True,check=True)
    translated=[x for line in output.stdout.splitlines() for x in json.loads(line)]
    if len(translated)!=len(texts):raise ValueError('Transliteration row count mismatch')
    lookup=dict(zip(texts,translated));aggregates={};edit_counts=Counter();out=[]
    for r in raw:
        f={k:r[k] for k in ['source1_entity_id','target_id','country','population','label']}
        for field in ['name','address']:
            ra,rb=r[field+'_a'],r[field+'_b'];na,nb=normalize(ra,BASE),normalize(rb,BASE)
            ta,tb=normalize(lookup[ra],BASE),normalize(lookup[rb],BASE)
            before=JaroWinkler.normalized_similarity(na,nb) if na and nb else 0.
            after=JaroWinkler.normalized_similarity(ta,tb) if ta and tb else 0.
            nonascii=any(ord(c)>127 for c in ra+rb)
            key=(r['country'],r['population'],field,nonascii)
            agg=aggregates.setdefault(key,Counter());agg.update({'pairs':1,'high_before':int(before>=.9),'high_after':int(after>=.9),'exact_before':int(bool(na) and na==nb),'exact_after':int(bool(ta) and ta==tb),'improved':int(after>before+.01),'worsened':int(after<before-.01)})
            f[field+'_transliterated_jw']=after
            if r['label']:
                # Minimal edit scripts characterize observed differences, not a generative causal model.
                for edit in Levenshtein.editops(na,nb):edit_counts[(r['country'],field,edit.tag)]+=1
                for kind,yes in {'case_only_equal':ra!=rb and ra.casefold()==rb.casefold(),
                     'token_reorder_equal':na!=nb and sorted(na.split())==sorted(nb.split()),
                     'one_side_blank':bool(na)!=bool(nb),
                     'nonascii_pair':nonascii,
                     'containment':bool(na and nb) and na!=nb and (na in nb or nb in na)}.items():
                    edit_counts[(r['country'],field,kind)]+=int(yes)
        out.append(f)
    numeric=[]
    for r in raw:
        aa,bb=r['address_a'],r['address_b'];na=re.findall(r'\d+',aa);nb=re.findall(r'\d+',bb);pa=set(re.findall(r'\b[0-9]{5,6}\b',aa));pb=set(re.findall(r'\b[0-9]{5,6}\b',bb))
        numeric.append({'country':r['country'],'population':r['population'],'both_numeric':bool(na and nb),'first_numeric_equal':bool(na and nb and na[0]==nb[0]),'numeric_conflict':bool(na and nb and not set(na)&set(nb)),'both_postal_like':bool(pa and pb),'postal_like_agreement':bool(pa&pb),'both_empty':not aa.strip() and not bb.strip()})
    numbers=pl.DataFrame(numeric)
    numbers.group_by('country','population').agg(pl.len(),*[pl.col(c).mean() for c in numbers.columns[2:]]).sort('country','population').write_csv(a.pairs/'numeric_diagnostics.csv')
    pl.DataFrame(out).write_parquet(a.pairs/'transliteration_features.parquet')
    summary={'scope':'Same frozen fold0 MORPH sample, local Foundation Any-Latin; Latin-ASCII. No external service. Optional parallel view only. Minimal-edit counts are descriptive, not true corruption-process frequencies.',
       'unique_strings':len(texts),'runtime_seconds':time.perf_counter()-start,
       'rates':[dict(zip(['country','population','field','nonascii'],key),**dict(v)) for key,v in aggregates.items()],
       'observed_edits':[dict(zip(['country','field','kind'],key),count=value) for key,value in edit_counts.items()]}
    dest.write_text(json.dumps(summary,indent=2));print(json.dumps({'unique_strings':len(texts),'runtime_seconds':summary['runtime_seconds']}))
if __name__=='__main__':main()
