"""Alternative numeric comparison view; preserve original numbers and features."""
from pathlib import Path
import json,re,unicodedata,hashlib
import duckdb,polars as pl
NAMES=['numeric_canonical_overlap','numeric_canonical_jaccard','numeric_canonical_conflict','numeric_leading_zero_rescue','address_first_numeric_canonical_equal','address_first_numeric_canonical_conflict']

def canonical_numbers(text):
    """Map Unicode decimal digits and leading zeros; never rewrite raw addresses."""
    return [(''.join(str(unicodedata.decimal(c)) for c in token).lstrip('0') or '0') for token in re.findall(r'\d+',text)]

def compare_numbers(a,b):
    left,right=canonical_numbers(a),canonical_numbers(b);aa,bb=set(left),set(right);shared=len(aa&bb);rawa,rawb=set(re.findall(r'\d+',a)),set(re.findall(r'\d+',b))
    return [float(shared>0),shared/len(aa|bb) if aa or bb else 0.,float(bool(aa and bb) and not shared),float(shared>0 and not rawa&rawb),float(bool(left and right) and left[0]==right[0]),float(bool(left and right) and left[0]!=right[0])]

def main():
    out=Path('artifacts/features/P4-NUMERIC-001');out.mkdir(parents=True,exist_ok=False);db=duckdb.connect('artifacts/audit.duckdb',read_only=True,config={'threads':2,'memory_limit':'2GB'});cur=db.execute("""SELECT p.source1_entity_id,p.target_id,s.a,t.a FROM read_parquet('outputs/oof/P4-A-001/pair_scores.parquet') p JOIN s1_normalized s ON s.entity_id=p.source1_entity_id JOIN targets_normalized t ON t.entity_id=p.target_id""");part=0
    while rows:=cur.fetchmany(25000):
        built=[(s,t,*compare_numbers(a,b)) for s,t,a,b in rows];pl.DataFrame(built,schema=['source1_entity_id','target_id',*NAMES],orient='row').with_columns(pl.col(NAMES).cast(pl.Float32)).write_parquet(out/f'part-{part:04}.parquet');part+=1
    (out/'manifest.json').write_text(json.dumps({'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'query_candidates_sha256':hashlib.sha256(Path('outputs/oof/P4-A-001/pair_scores.parquet').read_bytes()).hexdigest(),'features':NAMES,'policy':'Pure alternative per-record representation. No labels/statistics fitted; leading-zero equality is evidence only, not acceptance rule.'},indent=2));print(json.dumps({'parts':part}))
if __name__=='__main__':main()
