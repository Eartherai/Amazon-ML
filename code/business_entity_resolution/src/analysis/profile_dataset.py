"""Bounded full-corpus text profiling, without training or modifying raw data.

Run from the repository root with PYTHONPATH=code/business_entity_resolution.
Exact summaries and token document frequencies use all rows. Character ngrams
alone use a reproducible per-country SHA256-order sample, clearly labelled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from typing import Any

import duckdb

LOG = logging.getLogger(__name__)
FIELDS = ("business_name", "business_address")
SOURCES = tuple(f"{split}_source{s}" for split in ("train", "test") for s in (1, 2, 3))
QUANTILES = [.01, .05, .10, .25, .50, .75, .90, .95, .99]
SCRIPTS = {
    "latin": "\\p{Latin}", "devanagari": "\\p{Devanagari}",
    "bengali": "\\p{Bengali}", "gurmukhi": "\\p{Gurmukhi}",
    "gujarati": "\\p{Gujarati}", "oriya": "\\p{Oriya}",
    "tamil": "\\p{Tamil}", "telugu": "\\p{Telugu}",
    "kannada": "\\p{Kannada}", "malayalam": "\\p{Malayalam}",
    "arabic": "\\p{Arabic}", "cyrillic": "\\p{Cyrillic}", "han": "\\p{Han}",
}
CATEGORIES = {"letters": "L", "marks": "M", "numbers": "N", "punctuation": "P", "symbols": "S", "separators": "Z", "controls": "C"}


def sqlstr(value: Any) -> str:
    """Quote SQL literals; table and column names are fixed constants."""
    return "'" + str(value).replace("'", "''") + "'"


def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Return small aggregate results as serializable records."""
    cursor = con.execute(sql)
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def write_new(path: Path, data: Any) -> None:
    """Preserve existing evidence; every completed output is immutable."""
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def reserve_check(directory: Path) -> None:
    """Fail before further spill if the required disk reserve is threatened."""
    if shutil.disk_usage(directory).free < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; stop profiling to preserve disk reserve")


def rss_gib() -> float:
    """Normalize platform-specific getrusage peak RSS units."""
    scale = 1024**3 if sys.platform == "darwin" else 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def bucket_sql(column: str) -> str:
    return (f"CASE WHEN {column}=1 THEN '1' WHEN {column}=2 THEN '2' "
            f"WHEN {column}<=5 THEN '3-5' WHEN {column}<=10 THEN '6-10' "
            f"WHEN {column}<=100 THEN '11-100' WHEN {column}<=1000 THEN '101-1000' ELSE '1001+' END")


def duplicate_profile(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    result = []
    for column, representation in (("raw", "raw"), ("norm", "nfc_lower_alphanumeric_with_marks")):
        con.execute(f"CREATE OR REPLACE TEMP TABLE frequency AS SELECT country,{column} AS field_value,count(*) n FROM field_base GROUP BY country,{column}")
        values = rows(con, """SELECT country,count(*) distinct_values,
            count(*) FILTER(WHERE n>1) duplicate_groups,
            sum(n-1) duplicate_excess,max(n) largest_bucket,
            sum(n) FILTER(WHERE field_value='') blank_bucket_rows
            FROM frequency GROUP BY country ORDER BY country""")
        buckets = rows(con, f"SELECT country,{bucket_sql('n')} bucket,count(*) AS bucket_groups,sum(n) AS bucket_rows FROM frequency GROUP BY 1,2 ORDER BY 1,2")
        for value in values:
            value["representation"] = representation
            value["bucket_histogram"] = [x for x in buckets if x["country"] == value["country"]]
            result.append(value)
        con.execute("DROP TABLE frequency")
    return result


def vocabulary_profile(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Exact per-country document frequencies; no vocabulary fitting for models."""
    con.execute("""CREATE OR REPLACE TEMP TABLE vocabulary AS
        SELECT country,token,count(*)::BIGINT df
        FROM field_base,unnest(list_distinct(tokens)) AS u(token)
        GROUP BY country,token""")
    con.execute("""CREATE OR REPLACE TEMP TABLE positions AS
        SELECT country,token,sum(first_df)::BIGINT first_df,sum(last_df)::BIGINT last_df FROM (
          SELECT country,tokens[1] token,count(*) first_df,0 last_df FROM field_base WHERE len(tokens)>0 GROUP BY 1,2
          UNION ALL
          SELECT country,tokens[-1] token,0 first_df,count(*) last_df FROM field_base WHERE len(tokens)>0 GROUP BY 1,2
        ) GROUP BY 1,2""")
    counts = {x["country"]: x["n"] for x in rows(con, "SELECT country,count(*) n FROM field_base GROUP BY country")}
    by_country = []
    for country, count in sorted(counts.items()):
        where = f"country={sqlstr(country)}"
        summary = rows(con, f"SELECT count(*) unique_tokens,sum(df) document_token_incidence,count(*) FILTER(WHERE df=1) singleton_tokens FROM vocabulary WHERE {where}")[0]
        summary.update(country=country, documents=count)
        summary["top_tokens"] = rows(con, f"""SELECT v.token,v.df,ln(({count}+1.0)/(v.df+1.0))+1 idf,
            coalesce(p.first_df,0) first_df,coalesce(p.last_df,0) last_df
            FROM vocabulary v LEFT JOIN positions p USING(country,token)
            WHERE v.{where} ORDER BY df DESC,token LIMIT 60""")
        summary["df_histogram"] = rows(con, f"SELECT {bucket_sql('df')} bucket,count(*) tokens,sum(df) document_token_incidence FROM vocabulary WHERE {where} GROUP BY 1 ORDER BY 1")
        summary["idf_histogram"] = rows(con, f"SELECT floor(ln(({count}+1.0)/(df+1.0))+1)::INTEGER idf_floor,count(*) tokens,sum(df) document_token_incidence FROM vocabulary WHERE {where} GROUP BY 1 ORDER BY 1")
        for position in ("first", "last"):
            summary[f"top_{position}_tokens"] = rows(con, f"SELECT token,{position}_df df FROM positions WHERE {where} AND {position}_df>0 ORDER BY {position}_df DESC,token LIMIT 40")
        by_country.append(summary)
    con.execute("DROP TABLE vocabulary")
    con.execute("DROP TABLE positions")
    return {"scope": "all rows, per field/source/country; within-document repeated tokens counted once", "tokenizer": "NFC, lowercase, split on non-letter/mark/number; preserve combining marks", "idf_formula": "1 + ln((country_source_field_document_count+1)/(document_frequency+1)); descriptive only, not fitted model vocabulary", "by_country": by_country}


def sampled_ngrams(con: duckdb.DuckDBPyConnection, source: str, field: str, sample_n: int) -> list[dict[str, Any]]:
    result = []
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM {source} ORDER BY country").fetchall()]
    for country in countries:
        sample = con.execute(f"SELECT profile_norm({field}) FROM {source} WHERE country={sqlstr(country)} ORDER BY sha256(entity_id) LIMIT {sample_n}").fetchall()
        texts = [r[0] for r in sample]
        for n in (2, 3, 4, 5):
            freq: Counter[str] = Counter()
            for text in texts:
                freq.update(set(text[i:i+n] for i in range(max(0, len(text)-n+1))))
            result.append({"country": country, "n": n, "sample_rows": len(texts), "unique_sample_ngrams": len(freq), "top_ngrams": [{"ngram": key, "sample_df": value} for key, value in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:40]]})
    return result


def profile_field(con: duckdb.DuckDBPyConnection, source: str, field: str, sample_n: int, output: Path) -> dict[str, Any]:
    start = time.perf_counter()
    reserve_check(output)
    LOG.info("%s.%s: materialize bounded temporary text view", source, field)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE field_base AS
        SELECT country,{field} raw,profile_norm({field}) norm,
        regexp_extract_all(profile_norm({field}),'[\\p{{L}}\\p{{M}}\\p{{N}}]+') tokens
        FROM {source}""")
    category_terms = [f"sum(length(regexp_replace(raw,'[^\\p{{{cat}}}]','','g'))) {name}_characters" for name, cat in CATEGORIES.items()]
    script_terms = [f"count(*) FILTER(WHERE regexp_matches(raw,{sqlstr(pattern)})) script_{name}_rows" for name, pattern in SCRIPTS.items()]
    summary = rows(con, f"""SELECT country,count(*) AS "rows",
        count(*) FILTER(WHERE raw IS NULL) null_count,
        count(*) FILTER(WHERE raw='') empty_strings,
        count(*) FILTER(WHERE trim(raw)='') blank_after_trim,
        count(*) FILTER(WHERE raw<>trim(raw)) outer_whitespace_rows,
        count(*) FILTER(WHERE regexp_matches(raw,'\\s{{2,}}')) repeated_whitespace_rows,
        count(*) FILTER(WHERE lower(trim(raw)) IN ('na','n/a','null','none','nan')) textual_missing_markers,
        min(length(raw)) min_length,avg(length(raw)) mean_length,max(length(raw)) max_length,
        quantile_cont(length(raw),{QUANTILES}) length_quantiles,
        avg(len(tokens)) mean_tokens,max(len(tokens)) max_tokens,
        avg(len(regexp_extract_all(raw,'[0-9]+'))) mean_ascii_numeric_tokens,
        count(*) FILTER(WHERE regexp_matches(raw,'[0-9]')) ascii_digit_rows,
        count(*) FILTER(WHERE regexp_matches(raw,'[^\\x00-\\x7F]')) nonascii_rows,
        count(*) FILTER(WHERE regexp_matches(raw,'\\p{{M}}')) combining_mark_rows,
        count(*) FILTER(WHERE nfc_normalize(raw)<>raw) nfc_changes_rows,
        sum(length(raw)) total_characters,
        sum(length(regexp_replace(raw,'[^0-9]','','g'))) ascii_digit_characters,
        sum(length(regexp_replace(raw,'[^\\s]','','g'))) whitespace_characters,
        {','.join(category_terms+script_terms)}
        FROM field_base GROUP BY country ORDER BY country""")
    for item in summary:
        total = item["total_characters"] or 1
        item["character_category_ratios"] = {key: item[f"{key}_characters"] / total for key in CATEGORIES}
        item["length_quantiles"] = {f"p{int(q*100)}": value for q, value in zip(QUANTILES, item["length_quantiles"])}
    histograms = []
    for kind, expression in (("length", "length(raw)"), ("tokens", "len(tokens)"), ("ascii_numeric_tokens", "len(regexp_extract_all(raw,'[0-9]+'))")):
        values = rows(con, f'SELECT country,{expression} AS "value",count(*) n FROM field_base GROUP BY 1,2 ORDER BY 1,2')
        histograms.extend(dict(kind=kind, **value) for value in values)
    LOG.info("%s.%s: duplicate buckets", source, field)
    duplicate = duplicate_profile(con)
    LOG.info("%s.%s: exact token document frequencies", source, field)
    vocabulary = vocabulary_profile(con)
    punctuation = rows(con, """SELECT country,character,count(*) count FROM field_base,
        unnest(regexp_extract_all(raw,'[\\p{P}\\p{S}]')) AS u(character)
        GROUP BY 1,2 ORDER BY 1,3 DESC,2""")
    punct_rows = rows(con, """SELECT country,character,count(*) records FROM field_base,
        unnest(list_distinct(regexp_extract_all(raw,'[\\p{P}\\p{S}]'))) AS u(character)
        GROUP BY 1,2""")
    lookup = {(p["country"], p["character"]): p["records"] for p in punct_rows}
    for p in punctuation:
        p["records"] = lookup[p["country"], p["character"]]
    affixes = {}
    for kind, expression in (("prefix3", "left(norm,3)"), ("suffix3", "right(norm,3)")):
        affixes[kind] = rows(con, f"""SELECT country,affix,n FROM (
            SELECT country,{expression} affix,count(*) n,
            row_number() OVER(PARTITION BY country ORDER BY count(*) DESC,{expression}) rank
            FROM field_base WHERE norm<>'' GROUP BY 1,2) WHERE rank<=30 ORDER BY country,n DESC,affix""")
    con.execute("DROP TABLE field_base")
    ngrams = sampled_ngrams(con, source, field, sample_n)
    LOG.info("%s.%s done %.1fs", source, field, time.perf_counter()-start)
    return {"by_country": summary, "histograms": histograms, "duplicates": duplicate,
            "vocabulary": vocabulary, "punctuation": punctuation, "affixes": affixes,
            "sampled_ngrams": ngrams, "runtime_seconds": time.perf_counter()-start}


def render_markdown(report: dict[str, Any], destination: Path) -> None:
    """A compact audit with explicit scopes; machine-readable file has detail."""
    lines = ["# Full text and preprocessing profile", "", f"Generated {report['metadata']['completed_at']}. Exact counts cover all six source files; no labels are used in this profiler. Raw files and the audit database remain unchanged.", "", "## Scope and reproducibility", "", "- All missingness, lengths, token/numeric histograms, character categories, punctuation, token document frequencies, duplicate buckets and country slices are full-corpus exact aggregates.", "- Character 2/3/4/5-grams alone use the first 10,000 records per source/country in SHA256(entity_id) order. These are descriptive samples, not candidate recall measurements.", "- Profiling normalization is NFC + lowercase + replace non-letter/mark/number runs with spaces. It preserves Indic combining marks. No suffix removal, accent stripping, transliteration or inferred address correction is applied.", "- Token frequency dictionaries here are descriptive audit outputs. They must not become fitted features across validation boundaries.", "- Unicode/script coverage is a proxy, not a language detector. Country labels remain unrestricted strings.", "", "## Field and country summary", "", "| Source | Field | Country | Rows | Empty % | Outer whitespace % | Mean length | p95 | Mean tokens | Non-ASCII % | Mark % |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source in report["sources"]:
        for field, data in source["fields"].items():
            for item in data["by_country"]:
                n = item["rows"]
                lines.append(f"| {source['source']} | {field.removeprefix('business_')} | {item['country']} | {n:,} | {100*item['empty_strings']/n:.3f} | {100*item['outer_whitespace_rows']/n:.3f} | {item['mean_length']:.2f} | {item['length_quantiles']['p95']:.0f} | {item['mean_tokens']:.2f} | {100*item['nonascii_rows']/n:.2f} | {100*item['combining_mark_rows']/n:.2f} |")
    lines += ["", "## Collision and vocabulary interpretation", "", "Raw and normalized duplicate counts are per field/country, so common business words or addresses can produce large buckets without establishing an entity match. Full-record duplication from phase one is embedded unchanged in `phase1_exact`. Never collapse target IDs merely because text is identical. Rare tokens deserve retrieval/feature experiments; common tokens and legal suffixes need frequency weighting, not unconditional deletion.", "", "| Source | Field | Country | Raw distinct | Normalized distinct | Additional collapsed values | Unique tokens | Singleton-token share |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for source in report["sources"]:
        for field, data in source["fields"].items():
            for vocab in data["vocabulary"]["by_country"]:
                country = vocab["country"]
                matching = {x["representation"]: x for x in data["duplicates"] if x["country"] == country}
                raw = matching["raw"]["distinct_values"]
                norm = matching["nfc_lower_alphanumeric_with_marks"]["distinct_values"]
                lines.append(f"| {source['source']} | {field.removeprefix('business_')} | {country} | {raw:,} | {norm:,} | {raw-norm:,} | {vocab['unique_tokens']:,} | {100*vocab['singleton_tokens']/max(vocab['unique_tokens'],1):.2f}% |")
    meta = report["metadata"]
    lines += ["", "## Artifacts and runtime", "", "Machine-readable source checkpoints and aggregate: `artifacts/data_profile/PROFILE-001/`. Schema: `schema.json`. Reproduction command:", "", "```bash", "PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.analysis.profile_dataset --database artifacts/audit.duckdb --output-dir artifacts/data_profile/PROFILE-002 --markdown docs/DATA_PROFILE_002.md", "```", "", f"Elapsed {meta['runtime_seconds']:.1f}s; process peak RSS {meta['peak_rss_gib']:.3f} GiB. DuckDB memory budget 2 GB, 2 threads, spill ceiling 4 GB, >=8 GiB disk reserve checked. Existing phase-one data hashes are included per source. This is profiling only; no learned model, cloud job, test-label inference or external record enrichment."]
    with destination.open("x") as handle:
        handle.write("\n".join(lines)+"\n")


def additional_details(database: Path, output_dir: Path, markdown: Path) -> None:
    """Add postal-shape and punctuation unions without repeating costly scans.

    Named ASCII punctuation is derived from existing full-corpus document
    frequencies. A single combined scan per source supplies the exact union
    of punctuation records and country-level postal-shaped numeric patterns.
    """
    destination = output_dir / "additional_diagnostics.json"
    if destination.exists():
        raise FileExistsError(destination)
    start = time.perf_counter()
    report = json.loads((output_dir / "profile.json").read_text())
    con = duckdb.connect(str(database), read_only=True)
    con.execute("SET threads=2; SET memory_limit='2GB'; SET preserve_insertion_order=false")
    punctuation = {"ampersand": "&", "apostrophe_ascii": "'", "hyphen_ascii": "-", "comma": ",", "period": ".", "open_parenthesis": "(", "close_parenthesis": ")"}
    result: dict[str, Any] = {"schema_version": "1.0", "definitions": {
        "postal_like5_rows": "Address contains an ASCII word-boundary 5-digit token, optionally ZIP+4-shaped. This is a numeric shape proxy, not an identified/validated postal code.",
        "postal_like6_rows": "Address contains an ASCII word-boundary 6-digit token. This is a numeric shape proxy, not an identified/validated postal code.",
        "punctuation_rows": "Record contains at least one Unicode category P character; divide by source/country rows for record proportion.",
        "punctuation_or_symbol_rows": "Record contains Unicode category P or S; the union is counted directly, not summed from character frequencies.",
        "named_punctuation": "Exact ASCII character document frequencies derived from primary field profile; curly apostrophes, en/em dashes and other Unicode characters remain individually available in the primary punctuation table.",
        "parenthesis_any_rows": "Direct union of ASCII open/close parentheses, avoiding double counting records containing both.",
    }, "sources": []}
    for source in report["sources"]:
        name = source["source"]
        terms = []
        for field in FIELDS:
            for kind, regex in (("punctuation", "\\p{P}"), ("punctuation_or_symbol", "[\\p{P}\\p{S}]"), ("parenthesis_any", "[()]")):
                terms.append(f"count(*) FILTER(WHERE regexp_matches({field},{sqlstr(regex)})) {field}_{kind}_rows")
        terms += ["count(*) FILTER(WHERE regexp_matches(business_address,'\\b[0-9]{5}(-[0-9]{4})?\\b')) postal_like5_rows", "count(*) FILTER(WHERE regexp_matches(business_address,'\\b[0-9]{6}\\b')) postal_like6_rows"]
        LOG.info("Additional postal/punctuation diagnostics: %s", name)
        country_data = rows(con, f'SELECT country,count(*) AS "rows",'+','.join(terms)+f" FROM {name} GROUP BY country ORDER BY country")
        entry = {"source": name, "by_country": country_data, "postal_full_source_phase1": {"postal_like5_rows": source["phase1_exact"]["text_patterns"]["postal5_like"], "postal_like6_rows": source["phase1_exact"]["text_patterns"]["postal6_like"]}, "fields": {}}
        for field in FIELDS:
            values = []
            freq = {(x["country"], x["character"]): x["records"] for x in source["fields"][field]["punctuation"]}
            for item in country_data:
                value = {"country": item["country"], "rows": item["rows"]}
                for key, character in punctuation.items():
                    count = freq.get((item["country"], character), 0)
                    value[key] = {"records": count, "record_proportion": count/item["rows"]}
                for key in ("punctuation", "punctuation_or_symbol", "parenthesis_any"):
                    count = item[f"{field}_{key}_rows"]
                    value[key] = {"records": count, "record_proportion": count/item["rows"]}
                values.append(value)
            entry["fields"][field] = values
        for kind in ("postal_like5_rows", "postal_like6_rows"):
            assert sum(x[kind] for x in country_data) == entry["postal_full_source_phase1"][kind], (name, kind)
        result["sources"].append(entry)
    con.close()
    result["metadata"] = {"completed_at": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.perf_counter()-start, "peak_rss_gib": rss_gib(), "database_open_mode": "read_only", "threads": 2, "memory_limit": "2GB", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "postal_phase1_reconciliation": "12 exact full-source totals matched"}
    write_new(destination, result)
    write_new(output_dir / "additional_schema.json", {"schema_version": "1.0", "sources[].by_country": "country,rows,business_name/business_address punctuation union counts, postal_like5_rows/postal_like6_rows; all exact", "sources[].fields[field][]": "country,rows and named punctuation {records,record_proportion}; proportions are per-row, not per-character", "definitions": "precise shape and Unicode category semantics", "metadata": "runtime, process peak RSS, read-only connection, code hash, reconciliation"})
    lines = ["", "## Postal-shaped tokens and punctuation", "", "`additional_diagnostics.json` adds exact country-level postal-shape presence and punctuation row proportions. The 5-/6-digit matches are **numeric shape proxies, not identified or validated postal codes**. No country-specific postal dictionary or geocoding is used. Twelve source totals reconcile exactly with phase one. Named ampersand, ASCII apostrophe/hyphen/comma/period and parentheses proportions use exact character document frequencies; Unicode alternatives remain separately visible in the primary profile. Any-punctuation and either-parenthesis unions are counted directly.", "", "| Source | Country | 5-digit/ZIP+4-like address % | 6-digit address % | Name punctuation % | Address punctuation % |", "|---|---|---:|---:|---:|---:|"]
    for source in result["sources"]:
        for item in source["by_country"]:
            n = item["rows"]
            lines.append(f"| {source['source']} | {item['country']} | {100*item['postal_like5_rows']/n:.3f} | {100*item['postal_like6_rows']/n:.3f} | {100*item['business_name_punctuation_rows']/n:.2f} | {100*item['business_address_punctuation_rows']/n:.2f} |")
    lines += ["", f"Supplement scan elapsed {result['metadata']['runtime_seconds']:.1f}s; peak process RSS {result['metadata']['peak_rss_gib']:.3f} GiB."]
    with markdown.open("a") as handle:
        handle.write("\n".join(lines)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("artifacts/audit.duckdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_profile/PROFILE-001"))
    parser.add_argument("--phase1", type=Path, default=Path("docs/audit_evidence/audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/DATA_PROFILE.md"))
    parser.add_argument("--sample-rows", type=int, default=10000)
    parser.add_argument("--resume", action="store_true", help="Reuse completed immutable source checkpoints")
    parser.add_argument("--supplement-only", action="store_true", help="Add missing postal/punctuation details to a completed pre-supplement profile")
    args = parser.parse_args()
    if args.supplement_only:
        additional_details(args.database, args.output_dir, args.markdown)
        return
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Choose a new output directory, or --resume incomplete run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "profile.json").exists() or args.markdown.exists():
        raise FileExistsError("Completed reports are immutable; choose new output paths")
    reserve_check(args.output_dir)
    start = time.perf_counter()
    phase1 = json.loads(args.phase1.read_text())
    con = duckdb.connect(str(args.database), read_only=True)
    con.execute("SET threads=2; SET memory_limit='2GB'; SET preserve_insertion_order=false; SET max_temp_directory_size='4GB'")
    spill = args.output_dir / "spill"
    con.execute(f"SET temp_directory={sqlstr(spill)}")
    con.execute("CREATE TEMP MACRO profile_norm(x) AS trim(regexp_replace(lower(nfc_normalize(coalesce(x,''))), '[^\\p{L}\\p{M}\\p{N}]+', ' ', 'g'))")
    report: dict[str, Any] = {"schema_version": "1.0", "metadata": {"started_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "duckdb_version": duckdb.__version__, "database_open_mode": "read_only", "threads": 2, "memory_limit": "2GB", "max_spill": "4GB", "quantiles": QUANTILES, "ngram_sample_method": f"first {args.sample_rows} rows per source/country ORDER BY sha256(entity_id)", "script_note": "overlapping Unicode script proxies; not language identification", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "sources": []}
    schema = {"schema_version": "1.0", "root": {"metadata": "runtime/reproducibility/scope", "sources": "six source objects"}, "source": {"source": "train_source1 etc", "split": "train/test", "rows": "exact record count", "phase1_exact": "reused original full-audit record incl file hash, schema, full-record duplicate counts", "fields": "business_name/business_address objects"}, "field": {"by_country": "exact country aggregates: rows, missingness, character-category counts and ratios, overlapping script row counts, length quantiles p1/5/10/25/50/75/90/95/99", "histograms": "country, kind (length/tokens/ascii_numeric_tokens), value, n; exact", "duplicates": "country, representation, distinct_values, duplicate_groups/excess, largest_bucket, bucket_histogram; blank values included", "vocabulary": "by_country exact token DF, top first/last tokens, DF and natural-log smoothed IDF bins; descriptive only", "punctuation": "country, character, count (occurrences), records (DF); P and S Unicode categories", "affixes": "top normalized three-codepoint prefix/suffix per country; exact", "sampled_ngrams": "country,n,sample_rows,unique_sample_ngrams,top_ngrams with sample_df; sampled only", "runtime_seconds": "field wall time"}}
    if not (args.output_dir / "schema.json").exists():
        write_new(args.output_dir / "schema.json", schema)
    for source in SOURCES:
        checkpoint = args.output_dir / f"{source}.json"
        if checkpoint.exists():
            LOG.info("Reusing completed source %s", source)
            report["sources"].append(json.loads(checkpoint.read_text()))
            continue
        item = {"source": source, "split": source.split("_")[0], "rows": phase1["files"][source]["n"], "phase1_exact": phase1["files"][source], "fields": {}}
        for field in FIELDS:
            field_checkpoint = args.output_dir / f"{source}.{field}.json"
            if field_checkpoint.exists():
                item["fields"][field] = json.loads(field_checkpoint.read_text())
            else:
                item["fields"][field] = profile_field(con, source, field, args.sample_rows, args.output_dir)
                write_new(field_checkpoint, item["fields"][field])
        write_new(checkpoint, item)
        report["sources"].append(item)
    con.close()
    if spill.exists() and not any(spill.iterdir()):
        spill.rmdir()
    report["metadata"].update(completed_at=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter()-start, peak_rss_gib=rss_gib(), total_rows=sum(x["rows"] for x in report["sources"]))
    write_new(args.output_dir / "profile.json", report)
    render_markdown(report, args.markdown)
    additional_details(args.database, args.output_dir, args.markdown)
    LOG.info("Finished %d rows in %.1fs; peak RSS %.3f GiB", report["metadata"]["total_rows"], report["metadata"]["runtime_seconds"], report["metadata"]["peak_rss_gib"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
