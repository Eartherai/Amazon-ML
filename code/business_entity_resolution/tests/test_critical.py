from pathlib import Path
import pytest
from src.evaluation import entity_f05,evaluate,f05_from_counts
from src.io_utils import parse_id_list,read_lists,write_lists,validate_maps,merge_shards,shard_for

@pytest.mark.parametrize('truth,pred,expected',[
(set(),set(),1), (set(),{'x'},0), ({'x'},set(),0), ({'x'},{'y'},0),
({'a','b'},{'a','b','c'},5/7), ({'a','b'},{'a'},5/6),
({'a'},{'a','b'},5/9), ({'a'},{'a'},1)])
def test_hand_calculated_metric(truth,pred,expected):
    assert entity_f05(truth,pred)==pytest.approx(expected)
    assert f05_from_counts(len(truth&pred),len(pred),len(truth))==pytest.approx(expected)

def test_macro_not_micro():
    result=evaluate({'a':set(),'b':{'x','y'}},{'a':set(),'b':{'x','y','z'}})
    assert result['macro_f0_5']==pytest.approx(6/7)
    assert result['micro_precision']==pytest.approx(2/3)
    with pytest.raises(ValueError): evaluate({'a':set()}, {})

@pytest.mark.parametrize('bad',['S2-a,S2-a','S2-a,',' S2-a','S1-a','S2-a\t'])
def test_bad_ids(bad):
    with pytest.raises(ValueError): parse_id_list(bad)

def test_tsv_roundtrip_singletons(tmp_path):
    x={'S1-b':set(),'S1-a':{'S2-z','S3-x'}}; p=tmp_path/'out.tsv'
    write_lists(p,x,'matched_entity_ids'); assert read_lists(p)==x
    assert p.read_text().splitlines()[-1]=='S1-b\t'
    with pytest.raises(FileExistsError): write_lists(p,x,'matched_entity_ids')

def test_strict_membership_and_ids():
    with pytest.raises(ValueError): validate_maps({'S1-a'},{'S2-a'},{'S1-a':{'S2-a'}},{'S1-a':set()})
    with pytest.raises(ValueError): validate_maps({'S1-a'},set(),{'S1-a':set()},{'S1-a':{'S2-a'}})

def test_sharding_and_merge():
    a={'S1-a':{'S2-x'}}; b={'S1-b':set()}
    assert merge_shards([a,b],{'S1-a','S1-b'})==merge_shards([b,a],{'S1-a','S1-b'})
    with pytest.raises(ValueError): merge_shards([a,a],{'S1-a'})
    with pytest.raises(ValueError): merge_shards([a],{'S1-a','S1-b'})
    assert shard_for('S1-123',4)==shard_for('S1-123',4)

def test_no_na_coercion(tmp_path):
    p=tmp_path/'bad.tsv'; p.write_text('source1_entity_id\tmatched_entity_ids\nS1-a\t\nS1-a\t\n')
    with pytest.raises(ValueError): read_lists(p)

def test_validation_hash_is_stable_and_open_country():
    from src.build_validation import fold_for
    import hashlib
    key='S1-example'
    assert fold_for(key)==int(hashlib.sha256(('20260925|'+key).encode()).hexdigest()[:8],16)%5
    # Country labels are intentionally absent from the assignment function.
    for country in ['US','India','France','new-country','']:
        assert 0<=fold_for(key)<5

def test_unicode_marks_and_accent_views():
    from src.normalization import light,latin_accent_fold,compatible
    assert light('हाईटेक टेक्नोलॉजी')=='हाईटेक टेक्नोलॉजी'
    assert latin_accent_fold('हाईटेक café')=='हाईटेक cafe'
    assert compatible('ＡＣＭＥ & Sons')=='acme and sons'
    assert light('Société Française')=='société française'

def test_sql_python_light_agree():
    from src.audit_data import connect
    from src.normalization import light
    con=connect(':memory:')
    for value in ['हाईटेक टेक्नोलॉजी','హై కన్సల్టెంట్స్','সেভেন','Société Française','A & B','A\tB','']:
        assert con.execute('SELECT norm(?)',[value]).fetchone()[0]==light(value)

def test_official_validator_on_synthetic_fixture(tmp_path):
    import importlib.util
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('official_validator',root/'student_resource/utils/validate_submission.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for source,entity,country in [(1,'S1-french','France'),(2,'S2-target','France'),(3,'S3-other','UnseenCountry')]:
        (tmp_path/f'test_source{source}.tsv').write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\n'+f'{entity}\tName\tAddress\t{country}\n')
    m=tmp_path/'matches.tsv';c=tmp_path/'candidates.tsv'
    write_lists(m,{'S1-french':{'S2-target'}},'matched_entity_ids')
    write_lists(c,{'S1-french':{'S2-target'}},'candidate_entity_ids')
    errors,warnings=module.validate(str(m),str(c),str(tmp_path),check_ids=True)
    assert not errors and not warnings
