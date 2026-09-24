"""Candidate pilot contracts: complete pool, train-only fitting, empty keys, IDs."""
import duckdb
import pytest

from src.blocking.token_candidates import prepare_queries,generate_candidates,summarize_candidates


def fixture_connection():
    con=duckdb.connect(config={"threads":2,"memory_limit":"256MB"})
    con.execute("CREATE TABLE s1_normalized(entity_id VARCHAR,country VARCHAR,n VARCHAR,a VARCHAR)")
    con.execute("CREATE TABLE targets_normalized(entity_id VARCHAR,country VARCHAR,n VARCHAR,a VARCHAR)")
    con.execute("CREATE TABLE validation_folds(source1_entity_id VARCHAR,fold INT)")
    con.execute("CREATE TABLE target_ownership(target_id VARCHAR,owner_fold INT)")
    con.executemany("INSERT INTO s1_normalized VALUES (?,?,?,?)",[
        ("S1-a","UnseenCountry","rare alpha alpha","12 main road"),
        ("S1-b","France","",""),("S1-c","France","café hiver","3 rue hiver"),
        ("S1-locked","France","secret","11 street")])
    con.executemany("INSERT INTO validation_folds VALUES (?,?)",[("S1-a",0),("S1-b",0),("S1-c",0),("S1-locked",4)])
    con.executemany("INSERT INTO targets_normalized VALUES (?,?,?,?)",[
        ("S2-a","UnseenCountry","alpha rare","12 avenue"),
        ("S3-b","UnseenCountry","rare alpha alpha","15 elsewhere"),
        ("S2-empty","France","",""),("S3-c","France","café hiver","3 rue hiver"),
        ("S2-distractor","UnseenCountry","rare beta","17 main road"),
        ("S2-wrongcountry","France","rare alpha alpha","12 main road")])
    con.executemany("INSERT INTO target_ownership VALUES (?,?)",[
        ("S2-a",0),("S3-b",0),("S2-empty",1),("S3-c",0),("S2-distractor",1),("S2-wrongcountry",4)])
    return con


def test_full_pool_retrieval_ownership_empty_and_determinism():
    con=fixture_connection()
    queries=prepare_queries(con,500)
    config={"max_full_pool_token_df":1000,"max_query_tokens_per_field":3,
            "route_top_k_per_target_source":100,"rrf_constant":60}
    result=generate_candidates(con,config)
    assert result["query_count"]==3 and result["target_pool_count"]==6
    assert not any(row["entity_id"]=="S1-locked" for row in queries)
    pairs=con.execute("SELECT source1_entity_id,target_id FROM final_candidates").fetchall()
    assert ("S1-a","S2-a") in pairs and ("S1-a","S3-b") in pairs
    assert ("S1-a","S2-distractor") in pairs  # Full-pool distractors remain.
    assert ("S1-a","S2-wrongcountry") not in pairs
    assert not any(s1=="S1-b" for s1,_ in pairs)  # Empty keys never match.
    assert len(pairs)==len(set(pairs))
    assert con.execute("SELECT full_df,train_df FROM token_stats WHERE token='rare' AND field='name' AND country='UnseenCountry'").fetchone()==(3,1)
    assert con.execute("SELECT route_score FROM route_scores WHERE source1_entity_id='S1-a' AND target_id='S3-b' AND route='rare_name'").fetchone()[0]<3
    again=fixture_connection()
    prepare_queries(again,500);generate_candidates(again,config)
    assert con.execute("SELECT * FROM final_candidates ORDER BY 1,2").fetchall()==again.execute("SELECT * FROM final_candidates ORDER BY 1,2").fetchall()


def test_exact_candidate_oracle_and_country_source_slices():
    queries=[{"entity_id":"S1-a","country":"France"},{"entity_id":"S1-b","country":"UnseenCountry"}]
    truth={"S1-a":{"S2-x","S3-y"},"S1-b":set()}
    candidates={"S1-a":{"S2-x","S2-noise"},"S1-b":{"S2-noise"}}
    result=summarize_candidates(queries,truth,candidates,100)
    assert result["link_recall"]==.5
    assert result["oracle_macro_f0_5"]==pytest.approx((5/6+1)/2)
    assert result["oracle_singleton_f0_5"]==1
    assert result["singleton_candidate_free_rate"]==0
    assert result["candidate_pairs"]==3
    assert result["positive_entity_any_coverage"]==1
    assert result["positive_entity_all_coverage"]==0
    with pytest.raises(ValueError): summarize_candidates(queries,truth,{"S1-a":set()},100)


def test_fanout_and_route_caps_are_enforced():
    con=fixture_connection();prepare_queries(con,500)
    generate_candidates(con,{"max_full_pool_token_df":1,"max_query_tokens_per_field":1,
                             "route_top_k_per_target_source":1,"rrf_constant":60})
    assert con.execute("SELECT max(full_df) FROM selected_keys").fetchone()[0]<=1
    assert con.execute("SELECT max(route_rank) FROM capped_routes").fetchone()[0]<=1
    assert not con.execute("SELECT * FROM selected_keys WHERE token='rare'").fetchall()
