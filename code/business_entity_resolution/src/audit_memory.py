"""Measure one-source-at-a-time Polars buffer footprints and legal token endings."""
import argparse,gc,json,resource,time
from pathlib import Path
import polars as pl
from src.audit_data import connect,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.perf_counter();c=connect(a.database);out={}
    for split in ['train','test']:
        for source in [1,2,3]:
            name=f'{split}_source{source}'
            frame=pl.read_csv(a.data_dir/split/(name+'.tsv'),separator='\t',schema_overrides={k:pl.String for k in ['entity_id','business_name','business_address','country']},missing_utf8_is_empty_string=True,n_threads=4)
            out[name]={'rows':frame.height,'polars_estimated_buffer_bytes':frame.estimated_size(),'polars_schema':{k:str(v) for k,v in frame.schema.items()}}
            del frame;gc.collect()
            out[name]['legal_suffix_token_counts']=rows(c,f"SELECT regexp_extract(norm(business_name),'\\b(pvt ltd|private limited|limited|ltd|llc|inc|incorporated|corp|corporation|llp|sarl|sas|sa|gmbh)$') suffix,count(*) n FROM {name} GROUP BY 1 ORDER BY n DESC")
    report={'files':out,'scope':'Polars estimated_size reports materialized visible buffer bytes, not total process peak. Files loaded sequentially, not simultaneously. Exact token-boundary ending counts replace early suffix-string probes.',
        'elapsed_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
