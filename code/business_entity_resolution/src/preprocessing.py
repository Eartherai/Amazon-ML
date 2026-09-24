"""Composable, country-independent text views for entity-resolution experiments.

Raw values must remain available. None normalizes to an empty string, while
``representations`` retains the original value and an explicit null flag.
Lossy operations are optional views, never identity decisions or replacements
for raw data. Explicit maps are applied once, with longest-phrase precedence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import unicodedata

VERSION = "1.0.0"
OPERATIONS = frozenset({
    "nfc", "nfkc", "casefold", "whitespace", "punctuation_space",
    "punctuation_delete", "ampersand", "latin_accent_fold", "token_sort",
    "token_set", "number_format_join", "token_map",
})
BASE = ["nfc", "casefold", "punctuation_space", "whitespace"]
COMPATIBLE = ["nfkc", "casefold", "ampersand", "punctuation_space", "whitespace"]
VARIANTS: dict[str, list[str]] = {
    "raw": [],
    "light": BASE,
    "compatible": COMPATIBLE,
    "accent": COMPATIBLE + ["latin_accent_fold"],
    "compact": ["nfkc", "casefold", "ampersand", "punctuation_delete", "whitespace"],
    "sorted": COMPATIBLE + ["token_sort"],
    "deduped": COMPATIBLE + ["token_set"],
    "number_format": ["nfkc", "casefold", "number_format_join", "punctuation_space", "whitespace"],
    "legal_map": COMPATIBLE + ["token_map"],
    "address_map": COMPATIBLE + ["token_map"],
    "aggressive": ["nfkc", "casefold", "ampersand", "latin_accent_fold", "number_format_join",
                   "punctuation_delete", "whitespace", "token_set"],
}


def _latin_accent_fold(text: str) -> str:
    """Remove decomposed marks after Latin bases only; retain all other marks."""
    result: list[str] = []
    latin_base = False
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                result.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def _replace_tokens(text: str, token_map: Mapping[str, str]) -> str:
    """Perform non-recursive, exact, longest-first phrase substitutions."""
    prepared: dict[tuple[str, ...], list[str]] = {}
    for key, value in token_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Map keys and values must be strings")
        parts = tuple(key.split())
        if not parts or key != " ".join(parts):
            raise ValueError("Map keys must be nonempty, whitespace-normalized phrases")
        if len(parts) > 3:
            raise ValueError("Map keys are limited to three tokens")
        prepared[parts] = value.split()
    tokens = text.split()
    result: list[str] = []
    position = 0
    while position < len(tokens):
        for width in range(min(3, len(tokens) - position), 0, -1):
            key = tuple(tokens[position:position + width])
            if key in prepared:
                result.extend(prepared[key])
                position += width
                break
        else:
            result.append(tokens[position])
            position += 1
    return " ".join(result)


def normalize(text: str | None, operations: Sequence[str], field: str = "name",
              token_map: Mapping[str, str] | None = None) -> str:
    """Apply ordered operations without country assumptions or hidden state.

    ``punctuation_*`` treats all non-letter/mark/number characters as separators
    or deletions, except whitespace is retained for the delete variant. Thus
    symbols such as '+' are lossy too. ``number_format_join`` joins commas and
    periods between digits; it can conflate decimals and thousands and is unsafe
    as an identity key. It preserves slash/hyphen house-number distinctions.
    ``field`` is an explicit provenance label and must be name or address.
    Maps must already use the same normalized token representation as the stage
    where they are applied. No replacement is recursively re-applied.
    """
    if field not in {"name", "address"}:
        raise ValueError("field must be 'name' or 'address'")
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string or None")
    if isinstance(operations, str):
        raise TypeError("operations must be an ordered sequence, not one string")
    unknown = set(operations) - OPERATIONS
    if unknown:
        raise ValueError(f"Unknown normalization operations: {sorted(unknown)}")
    if "token_map" in operations and token_map is None:
        raise ValueError("token_map operation requires an explicit map, including an explicit empty map")
    result = "" if text is None else text
    for operation in operations:
        if operation in {"nfc", "nfkc"}:
            result = unicodedata.normalize(operation.upper(), result)
        elif operation == "casefold":
            result = result.casefold()
        elif operation == "whitespace":
            result = " ".join(result.split())
        elif operation == "punctuation_space":
            result = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in result)
        elif operation == "punctuation_delete":
            result = "".join(c for c in result if unicodedata.category(c)[0] in "LMN" or c.isspace())
        elif operation == "ampersand":
            result = result.replace("&", " and ")
        elif operation == "latin_accent_fold":
            result = _latin_accent_fold(result)
        elif operation == "token_sort":
            result = " ".join(sorted(result.split()))
        elif operation == "token_set":
            result = " ".join(sorted(set(result.split())))
        elif operation == "number_format_join":
            result = re.sub(r"(?<=\d)[,.](?=\d)", "", result)
        elif operation == "token_map":
            result = _replace_tokens(result, token_map or {})
    return result


def representations(text: str | None, field: str = "name",
                    token_map: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return raw, nullness, and independent optional views; omit absent maps."""
    result: dict[str, Any] = {"raw": text, "is_null": text is None}
    for key, operations in VARIANTS.items():
        if key == "raw" or (key.endswith("_map") and (token_map is None or key !=
            ("legal_map" if field == "name" else "address_map"))):
            continue
        result[key] = normalize(text, operations, field, token_map)
    return result


def config_hash(config: Mapping[str, Any], token_map: Mapping[str, str] | None = None) -> str:
    """Stable cache identity includes ordered config, explicit map, and Unicode."""
    payload = {"implementation_version": VERSION, "unicode_version": unicodedata.unidata_version,
               "config": config, "token_map": token_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_gib() -> float:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum / (1024 ** (3 if sys.platform == "darwin" else 2))


def _digest_values(values: Sequence[str]) -> str:
    """Length-delimited digest prevents separator or concatenation collisions."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _benchmark_worker(sample: Path, backend: str, operations: list[str]) -> dict[str, Any]:
    """Benchmark a fresh-process native implementation and exact Python parity."""
    import polars as pl
    frame = pl.read_parquet(sample)
    values = frame["text"].to_list()
    expected = [normalize(value, operations) for value in values]
    started = time.perf_counter()
    if backend == "python":
        actual = [normalize(value, operations) for value in values]
    elif backend == "polars":
        expression = pl.col("text").fill_null("")
        for operation in operations:
            if operation == "nfc":
                expression = expression.str.normalize("NFC")
            elif operation == "whitespace":
                # Python's str.split recognizes these control separators too.
                expression = expression.str.replace_all(r"[\s\x1c-\x1f]+", " ").str.strip_chars()
            elif operation == "punctuation_space":
                expression = expression.str.replace_all(r"[^\p{L}\p{M}\p{N}]", " ")
            else:
                raise ValueError(f"Native Polars benchmark operation unsupported: {operation}")
        actual = frame.select(expression.alias("result"))["result"].to_list()
    elif backend == "duckdb":
        import duckdb
        connection = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
        expression = "coalesce(text,'')"
        for operation in operations:
            if operation == "nfc":
                expression = f"nfc_normalize({expression})"
            elif operation == "whitespace":
                expression = f"trim(regexp_replace({expression}, '[\\p{{Z}}\\t\\n\\r\\f\\v\\x1c-\\x1f\\x85]+', ' ', 'g'))"
            elif operation == "punctuation_space":
                expression = f"regexp_replace({expression}, '[^\\p{{L}}\\p{{M}}\\p{{N}}]', ' ', 'g')"
            else:
                raise ValueError(f"Native DuckDB benchmark operation unsupported: {operation}")
        actual = [row[0] for row in connection.execute(
            f"SELECT {expression} FROM read_parquet(?) ORDER BY sample_row", [str(sample)]).fetchall()]
        connection.close()
    else:
        raise ValueError(backend)
    elapsed = time.perf_counter() - started
    mismatches = sum(a != b for a, b in zip(actual, expected))
    if len(actual) != len(expected):
        raise AssertionError("Benchmark row loss")
    return {"backend": backend, "operations": operations, "rows": len(values),
            "elapsed_seconds": elapsed, "rows_per_second": len(values) / elapsed,
            "peak_process_rss_gib": _rss_gib(), "parity_mismatches": mismatches,
            "output_sha256": _digest_values(actual), "reference_sha256": _digest_values(expected),
            "config_hash": config_hash({"operations": operations}),
            "timing_scope": "native computation and materialized output; source read and parity reference excluded; DuckDB connection startup included",
            "rss_scope": "entire fresh process, including source read and parity reference"}


def mine_and_benchmark(database: Path, output_dir: Path, audit_json: Path,
                       sample_per_stratum: int = 5000) -> None:
    """Mine proposal-only transforms from folds 1-3; run bounded local probes."""
    import duckdb
    import polars as pl
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    connection = duckdb.connect(str(database), read_only=True,
                               config={"threads": 2, "memory_limit": "2GB"})
    # Stable SHA256 sampling of pairs within country and target source; no IDs
    # become preprocessing features. All sampled S1 and targets have train ownership.
    query = """WITH sampled AS (
      SELECT p.source1_entity_id, p.target_id, f.country, f.fold,
             row_number() OVER (PARTITION BY f.country, left(p.target_id,2)
               ORDER BY sha256(p.source1_entity_id || '|' || p.target_id)) rank
      FROM positive_pairs p JOIN validation_folds f USING(source1_entity_id)
      WHERE f.fold IN (1,2,3)
    ), chosen AS (SELECT * FROM sampled WHERE rank <= ?)
    SELECT c.*, s.business_name source_name, t.business_name target_name,
           s.business_address source_address, t.business_address target_address
    FROM chosen c JOIN train_source1 s ON s.entity_id=c.source1_entity_id
    JOIN train_targets t ON t.entity_id=c.target_id
    ORDER BY c.country, left(c.target_id,2), c.rank"""
    records = connection.execute(query, [sample_per_stratum]).fetchall()
    columns = [column[0] for column in connection.description]
    connection.close()
    rows = [dict(zip(columns, record)) for record in records]
    if any(row["fold"] not in {1, 2, 3} for row in rows):
        raise AssertionError("Held-out sample leakage")
    pairs = pl.DataFrame(rows)
    pair_path = output_dir / "training_pair_sample.parquet"
    pairs.write_parquet(pair_path)
    mapping_counts: dict[str, Counter[tuple[str, str]]] = {"name": Counter(), "address": Counter()}
    mapping_entities: dict[str, dict[tuple[str, str], set[str]]] = {
        "name": defaultdict(set), "address": defaultdict(set)}
    token_exposures: dict[str, Counter[str]] = {"name": Counter(), "address": Counter()}
    mapping_context: dict[str, dict[tuple[str, str], Counter[str]]] = {
        "name": defaultdict(Counter), "address": defaultdict(Counter)}
    sample_values: list[dict[str, Any]] = []
    raw_values: dict[str, list[str]] = {"name": [], "address": []}
    for row in rows:
        for field in ("name", "address"):
            source = normalize(row[f"source_{field}"], COMPATIBLE)
            target = normalize(row[f"target_{field}"], COMPATIBLE)
            left_tokens, right_tokens = source.split(), target.split()
            token_exposures[field].update(set(right_tokens))
            # Sequence alignment with retained exact context, not free co-occurrence.
            matcher = difflib.SequenceMatcher(a=right_tokens, b=left_tokens, autojunk=False)
            shared = sum(block.size for block in matcher.get_matching_blocks())
            for tag, i, j, k, end in matcher.get_opcodes():
                if tag != "replace" or not (1 <= j-i <= 3 and 1 <= end-k <= 3) or shared < 1:
                    continue
                old, new = " ".join(right_tokens[i:j]), " ".join(left_tokens[k:end])
                # Numbers can be decisive identity evidence; never propose replacing them.
                if any(char.isdigit() for char in old + new) or old == new:
                    continue
                mapping = (old, new)
                mapping_counts[field][mapping] += 1
                mapping_entities[field][mapping].add(row["source1_entity_id"])
                mapping_context[field][mapping][f'{row["country"]}/{row["target_id"][:2]}'] += 1
            for role in ("source", "target"):
                value = row[f"{role}_{field}"]
                raw_values[field].append(value)
                sample_values.append({"sample_row": len(sample_values), "field": field,
                                      "role": role, "text": value})
    evidence: dict[str, list[dict[str, Any]]] = {}
    ambiguous: dict[str, list[dict[str, Any]]] = {}
    proposal_maps: dict[str, dict[str, str]] = {}
    for field in ("name", "address"):
        totals = Counter()
        alternatives: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (old, new), count in mapping_counts[field].items():
            totals[old] += count
            alternatives[old].append((new, count))
        evidence[field] = []
        proposal_maps[field] = {}
        for (old, new), count in mapping_counts[field].most_common():
            if count < 3:
                continue
            entities = len(mapping_entities[field][old, new])
            share = count / totals[old]
            item = {"target_phrase": old, "source_phrase": new, "pair_support": count,
                    "distinct_s1_support": entities, "conditional_replacement_share": share,
                    "alternative_replacements": len(alternatives[old]),
                    "target_token_pair_exposure": token_exposures[field].get(old),
                    "strata": dict(mapping_context[field][old, new]),
                    "direction": "target-to-S1", "automatic_approval": False}
            evidence[field].append(item)
            if count >= 10 and entities >= 8 and share >= 0.9 and len(old.split()) <= 2 and len(new.split()) <= 2:
                proposal_maps[field][old] = new
        ambiguous[field] = [{"target_phrase": old, "alternatives": sorted(values, key=lambda pair: -pair[1]),
                             "total_replacement_events": totals[old]}
                            for old, values in alternatives.items() if len(values) > 1 and totals[old] >= 3]
        ambiguous[field].sort(key=lambda item: -item["total_replacement_events"])
    (output_dir / "token_replacement_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    (output_dir / "ambiguous_token_mappings.json").write_text(json.dumps(ambiguous, ensure_ascii=False, indent=2))
    (output_dir / "proposal_maps_UNVALIDATED.json").write_text(json.dumps({
        "status": "research proposals only; not production defaults", "maps": proposal_maps,
        "warning": "Conditional replacement share is not precision. Alignment can pair unrelated words. Require held-out collision/precision evaluation before enabling."
    }, ensure_ascii=False, indent=2))
    # 20k positive pairs gives 80k field occurrences. Deterministically select up
    # to 100k, preserving native scripts and duplicate frequency in this population.
    sample_values = sample_values[:100000]
    sample_path = output_dir / "benchmark_input.parquet"
    pl.DataFrame(sample_values).write_parquet(sample_path)
    benchmark_results = []
    for operations in (["whitespace"], ["nfc", "punctuation_space", "whitespace"]):
        for backend in ("python", "duckdb", "polars"):
            completed = subprocess.run([sys.executable, "-m", "src.preprocessing", "benchmark-worker",
                "--sample", str(sample_path), "--backend", backend, "--operations", json.dumps(operations)],
                text=True, capture_output=True, check=True, env={**os.environ, "POLARS_MAX_THREADS": "2"})
            benchmark_results.append(json.loads(completed.stdout))
    if any(result["parity_mismatches"] for result in benchmark_results):
        raise AssertionError("Native parity failed; inspect benchmark before promotion")
    # All default representations run in a separate Python pass; this is not a
    # misleading claim that native lower() equals Unicode casefold().
    representation_start = time.perf_counter()
    represented = [representations(value) for value in pl.read_parquet(sample_path)["text"].to_list()]
    representation_seconds = time.perf_counter() - representation_start
    (output_dir / "benchmark.json").write_text(json.dumps({
        "native_results": benchmark_results,
        "all_unmapped_views_python": {"rows": len(represented), "elapsed_seconds": representation_seconds,
             "rows_per_second": len(represented)/representation_seconds, "peak_process_rss_gib": _rss_gib(),
             "views": list(represented[0]) if represented else []},
        "limitation": "Native benchmark covers exactly stated operations, not full casefold/accent/map pipeline. Native lower is not Unicode casefold."
    }, indent=2))
    source_audit = json.loads(audit_json.read_text())
    metadata = {"run_id": output_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_version": VERSION, "unicode_version": unicodedata.unidata_version,
        "python_version": platform.python_version(), "duckdb_version": duckdb.__version__,
        "polars_version": pl.__version__, "platform": platform.platform(),
        "config_hash": config_hash({"variants": VARIANTS, "sample_per_stratum": sample_per_stratum}),
        "module_sha256": _sha256(Path(__file__)), "audit_evidence_sha256": _sha256(audit_json),
        "data_sha256": {name: entry["sha256"] for name, entry in source_audit["files"].items()
                        if name.startswith("train") and "sha256" in entry},
        "sample_sha256": _sha256(pair_path), "benchmark_input_sha256": _sha256(sample_path),
        "pair_count": len(rows), "distinct_s1": pairs["source1_entity_id"].n_unique(),
        "folds": sorted(pairs["fold"].unique().to_list()), "benchmark_rows": len(sample_values),
        "strata": dict(Counter(f'{row["country"]}/{row["target_id"][:2]}' for row in rows)),
        "sampling": "Lowest SHA256(S1|target) per country/target-source, folds 1-3 only; stratified, not population-weighted",
        "database_read_only": True, "duckdb_threads": 2, "duckdb_memory_limit": "2GB",
        "elapsed_seconds": time.perf_counter()-start, "peak_process_rss_gib": _rss_gib(),
        "learned_maps_enabled_by_default": False,
        "map_evidence_counts": {field: len(items) for field, items in evidence.items()},
        "proposal_counts": {field: len(items) for field, items in proposal_maps.items()},
        "ambiguous_phrase_counts": {field: len(items) for field, items in ambiguous.items()}}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("benchmark-worker")
    worker.add_argument("--sample", required=True, type=Path)
    worker.add_argument("--backend", required=True, choices=["python", "duckdb", "polars"])
    worker.add_argument("--operations", required=True)
    mine = commands.add_parser("mine-and-benchmark")
    mine.add_argument("--database", required=True, type=Path)
    mine.add_argument("--output-dir", required=True, type=Path)
    mine.add_argument("--audit-json", type=Path, default=Path("docs/audit_evidence/audit.json"))
    mine.add_argument("--sample-per-stratum", type=int, default=6250)
    args = parser.parse_args()
    if args.command == "benchmark-worker":
        print(json.dumps(_benchmark_worker(args.sample, args.backend, json.loads(args.operations))))
    else:
        mine_and_benchmark(args.database, args.output_dir, args.audit_json, args.sample_per_stratum)


if __name__ == "__main__":
    main()
