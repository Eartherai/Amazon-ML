"""Behavioral safety contracts for optional preprocessing representations."""
import json
from pathlib import Path
import unicodedata

import pytest

from src.preprocessing import (BASE, COMPATIBLE, VARIANTS, config_hash,
                               normalize, representations)


@pytest.mark.parametrize("text", ["हाईटेक टेक्नोलॉजी", "హై కన్సల్టెంట్స్", "সেভেন ট্রেডিং", "தமிழ் நிறுவனம்"])
def test_indic_combining_marks_are_retained(text):
    expected = unicodedata.normalize("NFC", text)
    assert normalize(text, BASE) == expected
    assert normalize(text, COMPATIBLE + ["latin_accent_fold"]) == expected
    assert normalize(text, ["punctuation_delete"]) == text


def test_raw_and_null_semantics_are_separate():
    raw = "  Café\t& Sons  "
    result = representations(raw)
    assert result["raw"] == raw
    assert result["is_null"] is False
    assert result["compatible"] == "café and sons"
    missing = representations(None)
    empty = representations("")
    assert missing["raw"] is None and missing["is_null"]
    assert empty["raw"] == "" and not empty["is_null"]
    for key in missing.keys() - {"raw", "is_null"}:
        assert missing[key] == empty[key] == ""
    assert normalize(None, []) == ""
    assert normalize(raw, []) == raw


def test_unicode_compatibility_and_casefold_are_explicit():
    assert normalize("ＡＣＭＥ Straße", BASE) == "ａｃｍｅ strasse"
    assert normalize("ＡＣＭＥ Straße", COMPATIBLE) == "acme strasse"
    assert normalize("İẞ", ["casefold"]) == "i\u0307ss"
    assert normalize("e\u0301", ["nfc"]) == "é"


def test_latin_accents_preserve_non_latin_marks():
    assert normalize("Société Française & हिंदी", VARIANTS["accent"]) == "societe francaise and हिंदी"
    assert normalize("άλφα", ["latin_accent_fold"]) == "άλφα"
    assert normalize("\u0301é", ["latin_accent_fold"]) == "\u0301e"


def test_punctuation_operations_do_not_silently_conflate():
    assert normalize("A-B O'Reilly C++", BASE) == "a b o reilly c"
    assert normalize("A-B O'Reilly C++", VARIANTS["compact"]) == "ab oreilly c"
    assert normalize("a\tb\u00a0c\u2003d\x1ce", ["whitespace"]) == "a b c d e"
    assert normalize("R&D", COMPATIBLE) == "r and d"
    assert normalize("R&D", BASE) == "r d"
    assert normalize("Ａ＆Ｂ", COMPATIBLE) == "a and b"


def test_token_sort_retains_duplicates_and_set_is_separate():
    assert normalize("Beta Alpha Alpha", BASE + ["token_sort"]) == "alpha alpha beta"
    assert normalize("Beta Alpha Alpha", BASE + ["token_set"]) == "alpha beta"
    assert normalize("A B A", BASE) != normalize("A B", BASE)


def test_numeric_views_preserve_raw_and_do_not_merge_house_ranges():
    raw = "1,234 1.25 12/4 12-14 0012 १२,३४५"
    assert normalize(raw, ["number_format_join"]) == "1234 125 12/4 12-14 0012 १२३४५"
    assert representations(raw)["raw"] == raw
    assert representations(raw)["light"] == "1 234 1 25 12 4 12 14 0012 १२ ३४५"
    assert representations(raw)["number_format"] != representations(raw)["light"]


def test_maps_are_explicit_nonrecursive_longest_first_and_field_specific():
    mapping = {"co": "company", "co ltd": "limited company", "limited": "ltd", "pvt": "private"}
    assert normalize("Acme Co Ltd Pvt", COMPATIBLE + ["token_map"], token_map=mapping) == "acme limited company private"
    result = representations("Acme Co", token_map={"co": "company"})
    assert result["legal_map"] == "acme company" and "address_map" not in result
    result = representations("1 Main Rd", field="address", token_map={"rd": "road"})
    assert result["address_map"] == "1 main road" and "legal_map" not in result
    assert "legal_map" not in representations("Acme Co")
    assert normalize("Acme Co", COMPATIBLE + ["token_map"], token_map={}) == "acme co"
    assert normalize("x y z", ["token_map"], token_map={"x y": ""}) == "z"
    with pytest.raises(ValueError, match="explicit map"):
        normalize("Acme", ["token_map"])
    with pytest.raises(ValueError, match="nonempty"):
        normalize("Acme", ["token_map"], token_map={"": "a"})


def test_country_independent_and_field_explicit():
    # A country string is never an input to the normalization API.
    for country in ["US", "India", "France", "UnseenCountry", ""]:
        row = {"country": country, "name": "Étoile & Sons"}
        assert representations(row["name"])["compatible"] == "étoile and sons"
    with pytest.raises(ValueError, match="field"):
        normalize("text", BASE, field="country")
    with pytest.raises(TypeError):
        normalize(123, BASE)
    with pytest.raises(ValueError, match="Unknown"):
        normalize("text", ["delete_all_numbers"])
    with pytest.raises(TypeError):
        normalize("text", "casefold")


def test_long_strings_and_idempotence_of_recommended_views():
    raw = ("Café  हिंदी  ＡＣＭＥ & Sons\n" * 2000)
    result = representations(raw)
    assert result["raw"] == raw
    assert result["light"].count("हिंदी") == 2000
    for key in ("light", "compatible", "accent", "sorted", "deduped"):
        normalized = normalize(raw, VARIANTS[key])
        assert normalize(normalized, VARIANTS[key]) == normalized


def test_cache_hash_stable_but_sensitive_to_order_maps_and_field():
    a = {"operations": ["nfc", "casefold"], "field": "name"}
    b = {"field": "name", "operations": ["nfc", "casefold"]}
    assert config_hash(a, {"ltd": "limited", "co": "company"}) == config_hash(b, {"co": "company", "ltd": "limited"})
    assert config_hash(a) != config_hash({"operations": ["casefold", "nfc"], "field": "name"})
    assert config_hash(a) != config_hash({**a, "field": "address"})
    assert config_hash(a, {}) != config_hash(a, None)
    assert config_hash(a, {"ltd": "limited"}) != config_hash(a, {"ltd": "company"})


def test_native_backends_have_exact_parity_on_edge_cases(tmp_path):
    import polars as pl
    from src.preprocessing import _benchmark_worker
    values = [None, "", "a\tb\u00a0c\u2003d\x1ce\x85f", "हिंदी & Café", "e\u0301", "ＡＣＭＥ", "Straße", "1,234"]
    sample = tmp_path / "sample.parquet"
    pl.DataFrame({"sample_row": list(range(len(values))), "text": values}).write_parquet(sample)
    for operations in (["whitespace"], ["nfc", "punctuation_space", "whitespace"]):
        for backend in ("python", "duckdb", "polars"):
            result = _benchmark_worker(sample, backend, operations)
            assert result["parity_mismatches"] == 0
            assert result["output_sha256"] == result["reference_sha256"]
