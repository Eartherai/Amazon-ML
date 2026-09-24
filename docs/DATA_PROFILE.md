# Full text and preprocessing profile

Generated 2026-09-24T20:20:30.113041+00:00. Exact counts cover all six source files; no labels are used in this profiler. Raw files and the audit database remain unchanged.

## Scope and reproducibility

- All missingness, lengths, token/numeric histograms, character categories, punctuation, token document frequencies, duplicate buckets and country slices are full-corpus exact aggregates.
- Character 2/3/4/5-grams alone use the first 10,000 records per source/country in SHA256(entity_id) order. These are descriptive samples, not candidate recall measurements.
- Profiling normalization is NFC + lowercase + replace non-letter/mark/number runs with spaces. It preserves Indic combining marks. No suffix removal, accent stripping, transliteration or inferred address correction is applied.
- Token frequency dictionaries here are descriptive audit outputs. They must not become fitted features across validation boundaries.
- Unicode/script coverage is a proxy, not a language detector. Country labels remain unrestricted strings.

## Field and country summary

| Source | Field | Country | Rows | Empty % | Outer whitespace % | Mean length | p95 | Mean tokens | Non-ASCII % | Mark % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_source1 | name | India | 883,188 | 0.000 | 0.000 | 26.39 | 38 | 3.70 | 0.00 | 0.00 |
| train_source1 | name | US | 1,323,633 | 0.000 | 0.000 | 22.47 | 35 | 3.49 | 0.00 | 0.00 |
| train_source1 | address | India | 883,188 | 0.000 | 0.000 | 77.70 | 116 | 12.39 | 0.06 | 0.00 |
| train_source1 | address | US | 1,323,633 | 0.000 | 0.000 | 34.96 | 48 | 5.91 | 0.00 | 0.00 |
| train_source2 | name | India | 2,017,799 | 0.000 | 0.000 | 27.42 | 42 | 3.78 | 27.87 | 23.50 |
| train_source2 | name | US | 3,016,817 | 0.000 | 0.000 | 23.55 | 39 | 3.50 | 6.70 | 0.00 |
| train_source2 | address | India | 2,017,799 | 2.867 | 0.000 | 68.20 | 110 | 11.22 | 23.71 | 23.67 |
| train_source2 | address | US | 3,016,817 | 3.683 | 0.000 | 31.53 | 43 | 5.51 | 0.00 | 0.00 |
| train_source3 | name | India | 2,115,547 | 0.000 | 0.000 | 27.04 | 43 | 3.77 | 18.49 | 13.15 |
| train_source3 | name | US | 3,170,056 | 0.000 | 0.000 | 23.98 | 40 | 3.58 | 6.80 | 0.00 |
| train_source3 | address | India | 2,115,547 | 3.070 | 0.000 | 59.11 | 106 | 10.22 | 22.53 | 22.49 |
| train_source3 | address | US | 3,170,056 | 3.501 | 0.000 | 38.45 | 53 | 5.91 | 0.00 | 0.00 |
| test_source1 | name | France | 259,452 | 0.000 | 0.000 | 19.42 | 29 | 3.08 | 15.72 | 0.00 |
| test_source1 | name | India | 809,986 | 0.000 | 0.000 | 26.37 | 38 | 3.70 | 0.00 | 0.00 |
| test_source1 | name | US | 663,106 | 0.000 | 0.000 | 22.46 | 35 | 3.49 | 0.00 | 0.00 |
| test_source1 | address | France | 259,452 | 0.000 | 0.000 | 50.07 | 65 | 8.67 | 28.27 | 0.00 |
| test_source1 | address | India | 809,986 | 0.000 | 0.000 | 77.71 | 116 | 12.39 | 0.06 | 0.00 |
| test_source1 | address | US | 663,106 | 0.000 | 0.000 | 34.97 | 48 | 5.91 | 0.00 | 0.00 |
| test_source2 | name | France | 703,378 | 0.000 | 0.000 | 21.10 | 35 | 3.25 | 24.54 | 0.00 |
| test_source2 | name | India | 2,312,565 | 0.000 | 0.000 | 28.25 | 44 | 3.88 | 27.61 | 23.63 |
| test_source2 | name | US | 1,871,330 | 0.000 | 0.000 | 24.29 | 39 | 3.61 | 6.25 | 0.00 |
| test_source2 | address | France | 703,378 | 3.062 | 0.000 | 39.55 | 59 | 6.96 | 24.05 | 0.00 |
| test_source2 | address | India | 2,312,565 | 2.282 | 0.000 | 68.75 | 110 | 11.32 | 23.85 | 23.81 |
| test_source2 | address | US | 1,871,330 | 2.945 | 0.000 | 31.84 | 43 | 5.57 | 0.00 | 0.00 |
| test_source3 | name | France | 731,615 | 0.000 | 0.000 | 21.14 | 36 | 3.27 | 23.94 | 0.00 |
| test_source3 | name | India | 2,405,000 | 0.000 | 0.000 | 27.87 | 44 | 3.87 | 18.21 | 13.31 |
| test_source3 | name | US | 1,945,701 | 0.000 | 0.000 | 24.62 | 41 | 3.67 | 6.39 | 0.00 |
| test_source3 | address | France | 731,615 | 2.944 | 0.000 | 39.97 | 60 | 7.03 | 24.32 | 0.00 |
| test_source3 | address | India | 2,405,000 | 2.463 | 0.000 | 59.42 | 106 | 10.30 | 22.92 | 22.89 |
| test_source3 | address | US | 1,945,701 | 2.843 | 0.000 | 38.84 | 53 | 5.97 | 0.00 | 0.00 |

## Collision and vocabulary interpretation

Raw and normalized duplicate counts are per field/country, so common business words or addresses can produce large buckets without establishing an entity match. Full-record duplication from phase one is embedded unchanged in `phase1_exact`. Never collapse target IDs merely because text is identical. Rare tokens deserve retrieval/feature experiments; common tokens and legal suffixes need frequency weighting, not unconditional deletion.

| Source | Field | Country | Raw distinct | Normalized distinct | Additional collapsed values | Unique tokens | Singleton-token share |
|---|---|---|---:|---:|---:|---:|---:|
| train_source1 | name | India | 563,618 | 559,665 | 3,953 | 50,639 | 21.01% |
| train_source1 | name | US | 976,267 | 961,759 | 14,508 | 70,685 | 28.70% |
| train_source1 | address | India | 856,280 | 855,846 | 434 | 321,107 | 62.40% |
| train_source1 | address | US | 1,274,326 | 1,274,326 | 0 | 148,006 | 42.51% |
| train_source2 | name | India | 1,685,529 | 1,587,521 | 98,008 | 293,570 | 75.19% |
| train_source2 | name | US | 2,721,444 | 2,445,769 | 275,675 | 549,005 | 80.25% |
| train_source2 | address | India | 1,758,288 | 1,746,439 | 11,849 | 372,304 | 31.37% |
| train_source2 | address | US | 2,578,975 | 2,539,640 | 39,335 | 268,683 | 33.61% |
| train_source3 | name | India | 1,832,744 | 1,713,760 | 118,984 | 329,094 | 75.80% |
| train_source3 | name | US | 2,825,312 | 2,576,044 | 249,268 | 575,004 | 79.69% |
| train_source3 | address | India | 1,896,708 | 1,883,017 | 13,691 | 340,473 | 32.08% |
| train_source3 | address | US | 2,736,058 | 2,733,029 | 3,029 | 273,191 | 32.74% |
| test_source1 | name | France | 193,254 | 192,936 | 318 | 34,601 | 44.20% |
| test_source1 | name | India | 522,094 | 518,593 | 3,501 | 49,738 | 21.97% |
| test_source1 | name | US | 523,924 | 517,095 | 6,829 | 56,840 | 34.14% |
| test_source1 | address | France | 244,545 | 241,408 | 3,137 | 15,979 | 25.49% |
| test_source1 | address | India | 786,102 | 785,707 | 395 | 303,557 | 62.36% |
| test_source1 | address | US | 646,836 | 646,836 | 0 | 109,655 | 44.77% |
| test_source2 | name | France | 618,660 | 553,713 | 64,947 | 93,627 | 59.54% |
| test_source2 | name | India | 1,962,248 | 1,865,383 | 96,865 | 297,032 | 75.09% |
| test_source2 | name | US | 1,736,591 | 1,597,247 | 139,344 | 347,523 | 77.43% |
| test_source2 | address | France | 600,066 | 566,666 | 33,400 | 70,848 | 57.11% |
| test_source2 | address | India | 2,020,536 | 2,005,505 | 15,031 | 373,891 | 26.52% |
| test_source2 | address | US | 1,604,184 | 1,578,532 | 25,652 | 203,875 | 29.69% |
| test_source3 | name | France | 631,992 | 573,634 | 58,358 | 99,910 | 60.61% |
| test_source3 | name | India | 2,109,298 | 1,989,933 | 119,365 | 331,852 | 75.73% |
| test_source3 | name | US | 1,789,856 | 1,661,415 | 128,441 | 360,180 | 77.43% |
| test_source3 | address | France | 612,053 | 590,732 | 21,321 | 70,946 | 57.22% |
| test_source3 | address | India | 2,164,948 | 2,148,183 | 16,765 | 342,865 | 27.25% |
| test_source3 | address | US | 1,679,437 | 1,677,492 | 1,945 | 206,698 | 29.08% |

## Artifacts and runtime

Machine-readable source checkpoints and aggregate: `artifacts/data_profile/PROFILE-001/`. Schema: `schema.json`. Reproduction command:

```bash
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.analysis.profile_dataset --database artifacts/audit.duckdb --output-dir artifacts/data_profile/PROFILE-002 --markdown docs/DATA_PROFILE_002.md
```

Elapsed 824.2s; process peak RSS 2.809 GiB. DuckDB memory budget 2 GB, 2 threads, spill ceiling 4 GB, >=8 GiB disk reserve checked. Existing phase-one data hashes are included per source. This is profiling only; no learned model, cloud job, test-label inference or external record enrichment.

## Postal-shaped tokens and punctuation

`additional_diagnostics.json` adds exact country-level postal-shape presence and punctuation row proportions. The 5-/6-digit matches are **numeric shape proxies, not identified or validated postal codes**. No country-specific postal dictionary or geocoding is used. Twelve source totals reconcile exactly with phase one. Named ampersand, ASCII apostrophe/hyphen/comma/period and parentheses proportions use exact character document frequencies; Unicode alternatives remain separately visible in the primary profile. Any-punctuation and either-parenthesis unions are counted directly.

| Source | Country | 5-digit/ZIP+4-like address % | 6-digit address % | Name punctuation % | Address punctuation % |
|---|---|---:|---:|---:|---:|
| train_source1 | India | 0.270 | 0.000 | 12.33 | 100.00 |
| train_source1 | US | 10.820 | 0.125 | 26.42 | 100.00 |
| train_source2 | India | 1.073 | 0.000 | 26.60 | 97.13 |
| train_source2 | US | 10.131 | 1.387 | 38.55 | 96.32 |
| train_source3 | India | 0.998 | 0.000 | 27.91 | 96.93 |
| train_source3 | US | 10.174 | 1.329 | 39.26 | 96.50 |
| test_source1 | France | 0.387 | 0.013 | 16.23 | 100.00 |
| test_source1 | India | 0.255 | 0.000 | 12.25 | 100.00 |
| test_source1 | US | 10.818 | 0.119 | 26.32 | 100.00 |
| test_source2 | France | 0.496 | 0.018 | 28.17 | 96.91 |
| test_source2 | India | 1.045 | 0.000 | 25.34 | 97.72 |
| test_source2 | US | 10.340 | 1.425 | 37.12 | 97.06 |
| test_source3 | France | 0.511 | 0.016 | 28.24 | 97.04 |
| test_source3 | India | 0.960 | 0.000 | 26.51 | 97.54 |
| test_source3 | US | 10.407 | 1.370 | 37.83 | 97.16 |

Supplement scan elapsed 9.5s; peak process RSS 1.295 GiB.

## Verification

All 675 cross-checks passed across the six source files, 12 fields and 30 country/field slices. Histogram denominators, Unicode partitions, punctuation totals, duplicate buckets, vocabulary DF/IDF totals, quantile ordering and deterministic sample limits reconcile. The executed profiler source snapshot SHA256 matches profile metadata. Supplemental postal counts independently match all 12 phase-one source totals. See `integrity_checks.json`.
