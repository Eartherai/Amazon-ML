"""Compare final inference feature adapter against the saved measured training features."""
import duckdb,numpy as np,polars as pl
from src.models.sub001_features import NAMES,pair_features


def test_sampled_feature_parity():
    sampled=pl.scan_parquet('outputs/oof/P4-B-001/features/part-*.parquet').head(40).collect()
    numeric=pl.scan_parquet('artifacts/features/P4-NUMERIC-B-001/part-*.parquet').join(sampled.select('source1_entity_id','target_id').lazy(),on=['source1_entity_id','target_id']).collect()
    expected=sampled.join(numeric,on=['source1_entity_id','target_id'],validate='1:1')
    db=duckdb.connect('artifacts/audit.duckdb',read_only=True)
    ids=expected.select('source1_entity_id','target_id');db.execute('CREATE TEMP TABLE sample(source1_entity_id VARCHAR,target_id VARCHAR)');db.executemany('INSERT INTO sample VALUES (?,?)',ids.iter_rows())
    rows=db.execute("""SELECT s.entity_id,t.entity_id,s.country,t.country,s.n,s.a,t.n,t.a,coalesce(m.transliterated,t.n)
      FROM sample x JOIN s1_normalized s ON s.entity_id=x.source1_entity_id JOIN targets_normalized t ON t.entity_id=x.target_id
      LEFT JOIN read_parquet('artifacts/transliteration/TRANS-001/name_map.parquet') m ON m.n=t.n""").fetchall()
    bypair={(q,t):(qc,tc,qn,qa,tn,ta,tm)for q,t,qc,tc,qn,qa,tn,ta,tm in rows}
    mapping={tn:tm for _,_,_,_,_,_,tn,_,tm in rows if not tn.isascii()}
    for r in expected.to_dicts():
      qc,tc,qn,qa,tn,ta,tm=bypair[(r['source1_entity_id'],r['target_id'])]
      actual=pair_features(qn,qa,qc,tn,ta,tc,r['target_id'],r['name_retrieval_score'],r['address_retrieval_score'],r['name_reciprocal_rank'],r['address_reciprocal_rank'],r['retrieval_route_count'],mapping)
      np.testing.assert_allclose(actual,np.array([r[n] for n in NAMES],dtype=np.float32),atol=1e-6,rtol=1e-6)
