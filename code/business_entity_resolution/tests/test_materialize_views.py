"""Critical parity, losslessness and resumption tests for Parquet foundation."""
import json
from pathlib import Path
import polars as pl
import pytest
from src.preprocessing import normalize, BASE
from src.data.materialize_views import derive, regenerate, run


def test_unicode_parity_and_raw_preservation():
    values = [None, '', '  \t', 'Straße & SONS', 'İı ﬃ', 'Cafe\u0301', 'कंपनी सड़क १२३',
              'B+ Retail', '531/532-A', 'A\x1cB', 'ＡＢＣ', 'NA', 'x\u200dy', 'é Æ Œ']
    frame = pl.DataFrame({'entity_id':[str(i) for i in range(len(values))], 'business_name':values,
                          'business_address':values, 'country':['unseen']*len(values)})
    result=derive(frame)
    assert result.select(frame.columns).equals(frame)
    assert result['name_light'].to_list() == [normalize(x,BASE) for x in values]
    assert result['address_light'].to_list() == [normalize(x,BASE) for x in values]
    assert result['name_numeric_tokens'][6].to_list() == ['१२३']
    assert result['name_is_null'].sum() == 1
    assert result['name_is_blank'].sum() == 3
    assert regenerate('Straße & Co','compatible') == 'strasse and co'
    with pytest.raises(ValueError): regenerate('x','legal_map')


def test_roundtrip_resume_and_corruption(tmp_path):
    source=tmp_path/'raw'/'train'
    source.mkdir(parents=True)
    (source/'train_source1.tsv').write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tNA\t\tFrance\nS1-2\tStraße\t१२\tOther\nS1-3\tTest\tNULL\tUS\n')
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'rows_per_shard':2,'minimum_free_gib':0}))
    out=tmp_path/'processed'
    first=run(config,source.parent,out,['train_source1'])
    second=run(config,source.parent,out,['train_source1'])
    assert first['rows']==second['rows']==3
    assert len(first['sources'][0]['shards'])==2
    frame=pl.read_parquet(out/'train_source1'/'part-00000.parquet')
    assert frame['business_name'][0]=='NA' and frame['business_address'][0]==''
    shard=out/'train_source1'/'part-00000.parquet'
    shard.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Corrupted'):run(config,source.parent,out,['train_source1'])
