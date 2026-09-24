# Preprocessing implementation and evidence

Implemented 25 September 2026 IST in `code/business_entity_resolution/src/preprocessing.py`. The earlier `src/normalization.py` and original data are unchanged. There is no model training, cloud execution, or external business enrichment in this work.

## Recommended usage

Start with independent raw, light, and compatible views from `configs/preprocessing/PREP-001.json`. Do not replace raw columns or assume normalized equality proves identity. Exclude empty normalized keys from exact candidate blocks. Keep missingness as a separate feature.

```python
import json
from src.preprocessing import normalize, representations, config_hash

config = json.load(open("configs/preprocessing/PREP-001.json"))
raw = "Société & Sons"
views = {name: raw if name == "raw" else normalize(raw, operations, field="name")
         for name, operations in config["views"].items()}
cache_key = config_hash(config)

# Exploration helper returns all optional unmapped views, including clearly
# named lossy stress views. Production selects views through its config.
all_views = representations(raw)

# An explicit map is required; no dictionary is built into the normalizer.
mapped = normalize("1 Main Rd", ["nfc", "casefold", "punctuation_space",
                   "whitespace", "token_map"], field="address",
                   token_map={"rd": "road"})
```

`normalize(None, operations)` returns `""`. `representations(None)` retains `raw=None`, `is_null=True`, and empty strings for normalized views. An original empty string remains `raw=""`, `is_null=False`. Original strings, including whitespace and Unicode decomposition, are retained exactly in `raw`. No country parameter, country whitelist, ID feature, or test-derived dictionary exists.

`config_hash` includes the implementation version, Unicode database version, ordered operations/configuration, and explicitly supplied map. Dictionary insertion order does not affect it; changing operation order or a map value does. Include data version and field in the caller's cache configuration. Null and explicit empty maps remain distinct cache inputs.

## Operations and failure modes

| Operation | Behavior | Interpretation |
|---|---|---|
| `nfc` | Canonical Unicode composition | Conservative representation; preserves raw separately |
| `nfkc` | Compatibility composition, including full-width characters | May merge distinctions; separate compatible view |
| `casefold` | Full Unicode case folding, including `ß` to `ss` | More complete than `lower()`; not byte-equivalent to existing legacy light helper |
| `whitespace` | Python Unicode whitespace split/join | Includes tabs, NBSP, and control separators |
| `punctuation_space` | Keep Unicode letters, combining marks, numbers; replace all other characters with spaces | Preserves Indic marks; symbols and format controls also become spaces |
| `punctuation_delete` | Delete non-letter/mark/number symbols, retaining whitespace | Can merge `A-B` into `ab`; optional compact view only |
| `ampersand` | Replace `&` with `and` before punctuation removal | Token-view hypothesis; not semantic identity evidence |
| `latin_accent_fold` | Remove decomposed marks after Latin base letters only | Preserves Indic and Greek marks; accent collisions remain possible |
| `token_sort` | Sort all tokens; preserve duplicates | Order-invariant feature without multiplicity loss |
| `token_set` | Deduplicate and sort tokens | Loses multiplicity; separate optional ablation |
| `number_format_join` | Join comma or period only when between digits | `1.25` and `125` collide; slash/hyphen remain intact until any later punctuation operation |
| `token_map` | Explicit exact token/phrase map, longest match first, applied once | Requires reviewed map; never silently recursively rewrites |

Map keys contain 1–3 whitespace-normalized tokens and must match the representation at the point of application. Empty replacement values are possible but erase information and require validation. A map can still create conflicts or cycles across repeated calls; the normalizer intentionally does not infer a canonical direction from noisy data.

Configuration groups:

- `PREP-001.json`: raw/light/compatible, no fitted dictionaries.
- `PREP-002.json`: accent, compact, sorted, deduped, and number formatting ablations; disabled by default.
- `PREP-003.json`: explicit legal-name/address map hooks; empty default maps and a development-evaluation promotion gate.
- `PREP-099.json`: aggressive stress combination; explicitly unsafe and unapproved for production decisions.

No transliteration is implemented here. Latin-accent folding does not transliterate Hindi/Telugu/Bengali/Tamil. The existing ICU diagnostic remains a separate research route.

## Training-only replacement mining

`artifacts/preprocessing/PREP-001/` contains the completed evidence. It samples **25,000 labeled positive pairs from 24,786 S1 entities**, exclusively from frozen initial-training folds **1, 2, and 3**. There are 6,250 pairs in each India/US × S2/S3 stratum, selected by lowest SHA256(S1 ID | target ID). IDs select reproducible rows only; they are never preprocessing features. This stratified positive-pair sample is not population-weighted and excludes singletons, so it cannot estimate global prevalence or precision.

For each positive pair, the miner aligns compatible-view token sequences and records 1–3-token replacements only when at least one token matches elsewhere. It excludes replacements involving digits and does not infer replacements from unrestricted co-occurrence. Each retained evidence row records pair support, distinct S1 support, country/source strata, alternatives, and conditional replacement share. This share measures competing observed replacements, **not the probability that a substitution is correct**. Unchanged occurrences and negatives are not its denominator.

The run found:

| Evidence | Name | Address |
|---|---:|---:|
| Replacement rows with at least 3 observations | 138 | 264 |
| Phrases with multiple observed replacements | 54 | 150 |
| Proposals meeting frequency/share screen | 15 | 47 |

Useful observed patterns include `ltd → limited` (630 pairs), `l l c → llc` (140), `incorporated → inc` (84), `rd → road` (606), `st → street` (561), and `dr → drive` (547). These are training evidence, not approved mandatory substitutions. The screen deliberately retains ambiguity in its evidence files.

Two concrete hazards explain why automatic dictionary adoption is rejected:

- Both `corp → corporation` and `corporation → corp` have strong evidence. Applying them symmetrically can swap two equivalent forms instead of aligning them. An equivalence class needs a single reviewed representative and collision testing.
- `tennessee → tn` and `tn → tamil nadu` both appear. A globally applied dictionary could change a US state abbreviation into an Indian state. Context and geography matter, but no hard-coded US/India dictionary should become a mandatory pipeline step.

Alignment also proposes phrases containing textual `null` or location-specific aliases. Such values may reflect corruption rather than safe string equivalence. Proposal files are explicitly named `proposal_maps_UNVALIDATED.json`; **none is enabled by default**. Next evaluation should compare incremental positive rescue against distinct-entity collision costs on development fold 0, leave fold 4 locked, and consider features/alternative views instead of unconditional rewriting.

## Local throughput and parity

The benchmark uses **100,000 deterministic field occurrences**: source/target names and addresses from the 25,000 sampled pairs. These are occurrences, not 100,000 unique businesses. DuckDB runs read-only with 2 threads and a 2 GB memory limit; Polars workers use 2 threads. Source records remain private and local.

| Exactly benchmarked operations | Python rows/s | DuckDB rows/s | Polars rows/s |
|---|---:|---:|---:|
| Whitespace only | 2,457,853 | 871,501 | 3,200,610 |
| NFC + marks-preserving punctuation spacing + whitespace | 360,568 | 550,375 | 954,463 |

Every native result has **zero mismatches across all 100,000 strings**, with identical length-delimited output SHA256 hashes. Additional fixtures check null, decomposed accents, native Indic text, Unicode width, and unusual whitespace. DuckDB/Polars native `lower()` is **not** treated as equivalent to Python `casefold()`; the native benchmarks only claim equivalence for the exact operations shown above. Full normalization on deployment runtimes needs parity verification.

Computing the full Python helper bundle (raw/nullness plus eight optional unmapped views) took **3.63 seconds**, approximately **27,539 field occurrences/second**. If this short sample rate sustained across both fields of all 24.23M rows, the text-only projection would be roughly 29 minutes. That is a sizing estimate, not a full-data runtime promise: I/O, deduplication, serialization, cache strategy, field lengths, contention, and thermal limits differ. Production should compute only selected views, reuse repeated strings, and stream batches rather than retain all views for the entire corpus.

The complete mining/benchmark run took **9.85 seconds** with **1.44 GiB peak process RSS**. Separate benchmark workers used about 0.09–0.14 GiB peak RSS. Throughput timing excludes the Python parity-reference computation and initial source read; native materialization is included, and DuckDB connection startup is included. Peak RSS covers each full process. Results are single-run local probes, not statistically stable speed rankings.

## Reproduction and verification

Run from the repository root; the output path must be new and the command refuses to overwrite it:

```sh
PYTHONPATH=code/business_entity_resolution POLARS_MAX_THREADS=2 \
  .venv/bin/python -m src.preprocessing mine-and-benchmark \
  --database artifacts/audit.duckdb \
  --output-dir artifacts/preprocessing/PREP-NEW \
  --sample-per-stratum 6250

PYTHONPATH=code/business_entity_resolution POLARS_MAX_THREADS=2 \
  .venv/bin/python -m pytest code/business_entity_resolution/tests/test_preprocessing.py -q
```

**15 preprocessing tests pass.** They cover Indic marks, Latin accents, width/casefold, raw/null semantics, punctuation/ampersands, numeric hazards, explicit legal/address maps, longest-first nonrecursive substitutions, token multiplicity, unseen-country independence, long strings, cache stability, and native-engine parity. No libraries were added.

Key evidence files: `metadata.json`, `benchmark.json`, `training_pair_sample.parquet`, `benchmark_input.parquet`, `token_replacement_evidence.json`, `ambiguous_token_mappings.json`, `proposal_maps_UNVALIDATED.json`, and `data_lineage.json`. Metadata records source hashes, audit hash, module/config hashes, runtime versions, sample hashes, folds, strata, elapsed time and memory. Source hashes reference the completed audit; this run did not rehash the original multi-million-row TSV files. `data_lineage.json` also carries the label-file hash and frozen fold-manifest provenance.
