# Complete dataset audit

Generated 2026-09-25T01:12:00.585713+05:30.

All six source TSVs and the full ground truth were parsed. Exact aggregates cover every row. Near-duplicate probes and transliteration samples have explicitly limited scopes. No test labels were accessed or inferred. Official files were not modified.

## Core findings

- 24,229,173 source records; 2,206,821 train S1 entities and 1,732,544 test S1 entities.
- 7,638,365 positive links. Every ground-truth S1 and target ID exists; no duplicate S1 labels, repeated label IDs, or shared target owners.
- All positive links have equal country labels. Country remains an arbitrary string; France is present only in test.
- Raw S1 training names are ASCII. Many target variants use Indic scripts or diacritics. India requires more than Latin lexical matching.
- Do not deduplicate output IDs by text: S2/S3 contain repeated record text but each record ID remains a legitimate distinct target.
- Strict name AND address equality has excellent development precision but retrieves only 1.31% of positive links. It is not competitive.

## Inventory and memory

Raw textual columns are all VARCHAR strings; no numeric coercion, default pandas NA coercion, or country enumeration. Blank cells remain empty strings. Null counts and textual NA markers are separate. Disk sizes use MiB (2^20 bytes). The database compressed raw/label data to about 1.51 GiB before derived tables; the full audit process peaked near 3.14 GiB RSS on the Mac. These are actual process/database measurements, not estimated pandas memory. Later derived tables and probes increase artifact disk use. No 24M-row Python object dataframe was constructed.

| File | Rows | Disk MiB | Unique IDs | Unique raw names | Unique raw addresses | Raw duplicate excess | Normalized duplicate excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 200.34 | 2,206,821 | 1,539,229 | 2,130,606 | 0 | 0 |
| train_source2 | 5,034,616 | 466.63 | 5,034,616 | 4,402,009 | 4,337,262 | 25,873 | 67,708 |
| train_source3 | 5,285,603 | 480.37 | 5,285,603 | 4,651,609 | 4,632,765 | 18,860 | 50,425 |
| test_source1 | 1,732,544 | 166.91 | 1,732,544 | 1,238,867 | 1,677,483 | 0 | 1 |
| test_source2 | 4,887,273 | 485.86 | 4,887,273 | 4,311,041 | 4,224,784 | 22,641 | 56,112 |
| test_source3 | 5,082,316 | 482.56 | 5,082,316 | 4,521,929 | 4,456,436 | 16,293 | 41,790 |

Ground truth: 2,206,821 rows, 121.13 MiB. Source IDs are unique and prefixes agree with source files.

## Measured materialized memory

Polars estimated_size reports materialized visible buffer bytes, not total process peak. Files loaded sequentially, not simultaneously. Exact token-boundary ending counts replace early suffix-string probes.

| Source | Polars visible buffers MiB |
|---|---:|
| train_source1 | 191.92 |
| train_source2 | 447.43 |
| train_source3 | 460.21 |
| test_source1 | 160.30 |
| test_source2 | 467.21 |
| test_source3 | 463.17 |

Sequential memory-audit peak process RSS: 1.687 GiB, elapsed 8.17s. Raw source content was read independently by both DuckDB and Polars and row counts agreed.

## Missingness and empties

Each entry is count (percentage). `null_count` is parser nulls; `empty` is zero-length text; `blank` includes whitespace-only values; markers are literal NA/N/A/null/none/nan strings. Literal markers are not automatically replaced.

| File | Field | Null | Empty | Blank after trim | Textual markers |
|---|---|---:|---:|---:|---:|
| train_source1 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source1 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source1 | business_address | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source1 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source2 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source2 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 6 (0.000%) |
| train_source2 | business_address | 0 (0.000%) | 168,967 (3.356%) | 168,967 (3.356%) | 0 (0.000%) |
| train_source2 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source3 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| train_source3 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 18 (0.000%) |
| train_source3 | business_address | 0 (0.000%) | 175,916 (3.328%) | 175,916 (3.328%) | 0 (0.000%) |
| train_source3 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source1 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source1 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source1 | business_address | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source1 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source2 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source2 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 49 (0.001%) |
| test_source2 | business_address | 0 (0.000%) | 129,408 (2.648%) | 129,408 (2.648%) | 0 (0.000%) |
| test_source2 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source3 | entity_id | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| test_source3 | business_name | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 61 (0.001%) |
| test_source3 | business_address | 0 (0.000%) | 136,098 (2.678%) | 136,098 (2.678%) | 0 (0.000%) |
| test_source3 | country | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

## Country distribution

| File | Country | Rows | Share |
|---|---|---:|---:|
| train_source1 | US | 1,323,633 | 59.979% |
| train_source1 | India | 883,188 | 40.021% |
| train_source2 | US | 3,016,817 | 59.921% |
| train_source2 | India | 2,017,799 | 40.079% |
| train_source3 | US | 3,170,056 | 59.975% |
| train_source3 | India | 2,115,547 | 40.025% |
| test_source1 | India | 809,986 | 46.751% |
| test_source1 | US | 663,106 | 38.274% |
| test_source1 | France | 259,452 | 14.975% |
| test_source2 | India | 2,312,565 | 47.318% |
| test_source2 | US | 1,871,330 | 38.290% |
| test_source2 | France | 703,378 | 14.392% |
| test_source3 | India | 2,405,000 | 47.321% |
| test_source3 | US | 1,945,701 | 38.284% |
| test_source3 | France | 731,615 | 14.395% |

Test S1 shifts from about 60% US/40% India to 38.3% US/46.8% India/15.0% France. Train-only overall validation therefore overweights US relative to test. No France quality estimate is available from labels.

## Field lengths

Unicode codepoint lengths; not byte lengths or tokenizer counts. Quantiles are exact across each full source file.

| File | Field | Mean | Min | p1 | p25 | Median | p75 | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_source1 | business_name | 24.03 | 3.00 | 8.00 | 18.00 | 24.00 | 30.00 | 37.00 | 42.00 | 105.00 |
| train_source1 | business_address | 52.07 | 11.00 | 22.00 | 33.00 | 41.00 | 70.00 | 103.00 | 124.00 | 256.00 |
| train_source2 | business_name | 25.10 | 2.00 | 8.00 | 19.00 | 25.00 | 31.00 | 40.00 | 48.00 | 104.00 |
| train_source2 | business_address | 46.23 | 0.00 | 0.00 | 30.00 | 37.00 | 61.00 | 96.00 | 118.00 | 249.00 |
| train_source3 | business_name | 25.20 | 2.00 | 6.00 | 18.00 | 25.00 | 31.00 | 42.00 | 50.00 | 123.00 |
| train_source3 | business_address | 46.71 | 0.00 | 0.00 | 35.00 | 42.00 | 54.00 | 91.00 | 115.00 | 240.00 |
| test_source1 | business_name | 23.84 | 3.00 | 8.00 | 18.00 | 24.00 | 29.00 | 36.00 | 42.00 | 92.00 |
| test_source1 | business_address | 57.21 | 11.00 | 24.00 | 36.00 | 50.00 | 74.00 | 105.00 | 126.00 | 268.00 |
| test_source2 | business_name | 25.70 | 2.00 | 8.00 | 19.00 | 25.00 | 32.00 | 42.00 | 49.00 | 102.00 |
| test_source2 | business_address | 50.41 | 0.00 | 0.00 | 32.00 | 43.00 | 67.00 | 99.00 | 120.00 | 269.00 |
| test_source3 | business_name | 25.66 | 2.00 | 6.00 | 19.00 | 25.00 | 32.00 | 42.00 | 50.00 | 103.00 |
| test_source3 | business_address | 48.74 | 0.00 | 0.00 | 35.00 | 43.00 | 59.00 | 94.00 | 117.00 | 267.00 |

## Characters, scripts and transliteration

No language detector is assumed: Unicode block coverage is a script proxy, and Latin text cannot distinguish language or romanized Indic content. Counts can overlap when records mix scripts. Name and address non-ASCII counts are raw-field diagnostics.

| File | Non-ASCII names | Non-ASCII addresses | Names with digits | Addresses with digits |
|---|---:|---:|---:|---:|
| train_source1 | 0 | 554 | 35,585 | 2,129,784 |
| train_source2 | 764,608 | 478,453 | 255,636 | 4,563,675 |
| train_source3 | 606,737 | 476,588 | 270,847 | 4,799,297 |
| test_source1 | 40,789 | 73,800 | 20,207 | 1,660,706 |
| test_source2 | 928,158 | 720,665 | 190,009 | 4,529,243 |
| test_source3 | 737,515 | 729,222 | 202,470 | 4,700,823 |

**train_source1 script coverage:** devanagari=0, bengali=0, gurmukhi=0, gujarati=0, oriya=0, tamil=0, telugu=0, kannada=0, malayalam=0.

**train_source2 script coverage:** devanagari=479,500, bengali=54,757, gurmukhi=12,114, gujarati=54,763, oriya=12,207, tamil=60,220, telugu=65,393, kannada=66,637, malayalam=30,640.

**train_source3 script coverage:** devanagari=394,374, bengali=45,061, gurmukhi=10,319, gujarati=45,006, oriya=9,559, tamil=49,864, telugu=52,500, kannada=55,457, malayalam=24,342.

**test_source1 script coverage:** devanagari=0, bengali=0, gurmukhi=0, gujarati=0, oriya=0, tamil=0, telugu=0, kannada=0, malayalam=0.

**test_source2 script coverage:** devanagari=550,530, bengali=63,445, gurmukhi=14,133, gujarati=63,211, oriya=14,046, tamil=69,865, telugu=75,238, kannada=78,006, malayalam=35,766.

**test_source3 script coverage:** devanagari=454,885, bengali=51,991, gurmukhi=11,454, gujarati=52,233, oriya=11,287, tamil=57,708, telugu=60,471, kannada=64,518, malayalam=28,396.

First 10000 labeled positive pairs sorted by SHA256(S1|target) among non-ASCII target business names. Conditional sample, NOT an overall rate or candidate recall. No negatives tested in this probe.

Engine: macOS Foundation StringTransform Any-Latin; Latin-ASCII (ICU), runtime macOS 27.0; Linux ICU parity is unverified. Runtime for transformation/scoring: 0.79s.

| Country | Sample pairs | Raw/light JW >= .90 | Transliterated JW >= .90 | Raw JW < .70 rescued to >= .90 | Exact after transliteration |
|---|---:|---:|---:|---:|---:|
| India | 6,732 | 16.49% | 29.78% | 10.65% | 12.36% |
| US | 3,268 | 78.61% | 85.99% | 0.12% | 62.48% |

This is positive-similarity evidence, not measured retrieval recall or precision. Next test must include realistic distractors and transliteration collisions. Foundation is a local generic transform, not external enrichment; Linux ICU output parity remains to be tested.

## Legal endings and numeric/postal shapes

The authoritative ending counts use a token boundary and a documented generic suffix list. These are lexical proxies, not verified legal forms. Token-boundary counts in memory.json supersede the initial suffix-string probes. Do not use these counts to strip legal identity information automatically. Numeric token counts use digit runs; postals are only five/six-digit shapes, not real geocoded codes.

**train_source1**

- Ending counts (token boundary): private limited=431,869, llc=355,736, inc=238,306, pvt ltd=121,461, limited=90,046, llp=39,725, corp=33,761, ltd=27,134, corporation=13,540, incorporated=3.
- Address numeric-token distribution (tokens:records): 0:77,037, 1:1,278,852, 2:581,911, 3:164,991, 4:61,786, 5:25,182, 6:10,073, 7:4,065, 8:1,577, 9:626, 10:316, 11:155, 12:100, 13:51, 14:31, 15:21, 16:11, 17:11, 18:6, 19:7, 20:4, 21:2, 22:2, 24:1, 27:2, 28:1.
- Postal5-like: 145,601; postal6-like: 1,656.

**train_source2**

- Ending counts (token boundary): llc=410,508, inc=319,146, ltd=248,769, private limited=218,862, limited=203,764, corp=117,497, pvt ltd=99,403, corporation=54,370, llp=49,827, incorporated=23,165, sa=61, gmbh=11, sas=7, sarl=1.
- Address numeric-token distribution (tokens:records): 0:470,941, 1:2,824,921, 2:1,053,842, 3:413,588, 4:164,704, 5:64,261, 6:25,369, 7:10,048, 8:3,829, 9:1,482, 10:729, 11:365, 12:200, 13:112, 14:74, 15:51, 16:21, 17:27, 18:15, 19:11, 20:11, 21:5, 22:4, 24:1, 27:2, 28:3.
- Postal5-like: 327,300; postal6-like: 41,851.

**train_source3**

- Ending counts (token boundary): llc=449,544, inc=336,175, private limited=301,681, ltd=247,683, limited=237,742, pvt ltd=123,387, corp=115,490, llp=59,439, corporation=49,627, incorporated=19,241, sa=123, gmbh=10, sas=7.
- Address numeric-token distribution (tokens:records): 0:486,306, 1:2,853,699, 2:1,299,731, 3:412,206, 4:145,963, 5:53,934, 6:20,687, 7:7,835, 8:2,900, 9:1,173, 10:522, 11:248, 12:155, 13:76, 14:52, 15:38, 16:27, 17:11, 18:6, 19:12, 20:6, 21:3, 22:4, 23:1, 24:1, 26:3, 28:3, 29:1.
- Postal5-like: 343,630; postal6-like: 42,140.

**test_source1**

- Ending counts (token boundary): private limited=396,230, llc=178,364, inc=119,196, pvt ltd=110,856, limited=82,226, sarl=73,480, sas=52,276, llp=36,037, ltd=23,783, corp=19,782, sa=12,754, corporation=9,993, gmbh=3, incorporated=1.
- Address numeric-token distribution (tokens:records): 0:71,838, 1:1,016,131, 2:408,777, 3:140,432, 4:56,299, 5:23,270, 6:9,289, 7:3,810, 8:1,441, 9:645, 10:256, 11:142, 12:92, 13:46, 14:24, 15:15, 16:11, 17:7, 18:5, 19:2, 20:3, 21:1, 22:3, 23:1, 24:1, 26:1, 28:1, 31:1.
- Postal5-like: 74,808; postal6-like: 826.

**test_source2**

- Ending counts (token boundary): private limited=260,039, limited=256,146, ltd=255,398, llc=249,689, inc=200,416, pvt ltd=119,495, sarl=112,380, corp=96,050, sas=79,142, llp=65,214, corporation=41,155, sa=28,368, incorporated=12,994, gmbh=13.
- Address numeric-token distribution (tokens:records): 0:358,030, 1:2,784,019, 2:997,528, 3:438,128, 4:184,320, 5:75,432, 6:29,795, 7:11,738, 8:4,576, 9:1,852, 10:784, 11:478, 12:248, 13:149, 14:80, 15:27, 16:26, 17:23, 18:6, 19:12, 20:9, 21:5, 22:3, 23:3, 26:1, 31:1.
- Postal5-like: 221,147; postal6-like: 26,789.

**test_source3**

- Ending counts (token boundary): private limited=354,737, limited=296,756, llc=269,998, ltd=256,976, inc=209,008, pvt ltd=146,551, sarl=116,201, corp=93,396, sas=81,918, llp=77,123, corporation=38,060, sa=29,166, incorporated=10,706, gmbh=14.
- Address numeric-token distribution (tokens:records): 0:381,493, 1:2,884,294, 2:1,142,883, 3:413,156, 4:159,114, 5:62,443, 6:23,514, 7:9,101, 8:3,504, 9:1,420, 10:689, 11:258, 12:172, 13:105, 14:54, 15:34, 16:26, 17:16, 18:11, 19:9, 20:2, 22:8, 23:2, 24:2, 26:4, 28:1, 29:1.
- Postal5-like: 229,317; postal6-like: 26,777.

## Exact and near duplicates

Raw duplicate key=(name,address,country). Normalized key uses NFC+lower and Unicode letters, marks and numbers. Excess is rows minus distinct keys; ID uniqueness remains separate. S1 train has zero exact or normalized full-text duplicates. S1 test has one normalized duplicate group (two IDs): both must still be output. Raw names alone repeat frequently.

Exact count for adjacent records sorted by country, normalized name, normalized address, ID within each source; JW>=0.90 both fields, excluding identical normalized pairs and blank fields. Lower-bound diagnostic, not exhaustive fuzzy clustering.

| File | Adjacent pairs evaluated | Near pairs |
|---|---:|---:|
| train_source1 | 2,206,819 | 57 |
| train_source2 | 5,034,614 | 296,320 |
| train_source3 | 5,285,601 | 253,827 |
| test_source1 | 1,732,541 | 669 |
| test_source2 | 4,887,270 | 304,596 |
| test_source3 | 5,082,313 | 255,781 |

No claim of exhaustive fuzzy duplicate discovery: that would require a much broader blocking study. This is a deterministic lower-bound diagnostic over every source.

## Ground-truth topology

| Matches per S1 | Entities | Share |
|---|---:|---:|
| 0 | 123,247 | 5.585% |
| 1 | 119,157 | 5.399% |
| 2 | 375,212 | 17.002% |
| 3 | 530,841 | 24.055% |
| 4 | 484,115 | 21.937% |
| 5 | 321,957 | 14.589% |
| 6 | 164,868 | 7.471% |
| 7 | 63,968 | 2.899% |
| 8 | 18,680 | 0.846% |
| 9 | 4,205 | 0.191% |
| 10 | 534 | 0.024% |
| 11 | 37 | 0.002% |

Grouped: 0=5.585%; 1=5.399%; 2=17.002%; 3+=72.013%.

| Source relationship | Entities | Share of all S1 |
|---|---:|---:|
| no matches | 123,247 | 5.585% |
| S3 only | 164,498 | 7.454% |
| S2 only | 143,029 | 6.481% |
| both | 1,776,047 | 80.480% |

All 7,638,365 labeled targets have exactly one S1 owner. This supports **target-only** exclusivity experiments; S1 has many targets so one-to-one assignment is inappropriate. Training targets without any labeled owner number 2,681,854. They are distractors under the supplied exhaustive-label interpretation.

## Country differences

| Country | S1 | Mean matches | Singletons | One | Two | Three+ |
|---|---:|---:|---:|---:|---:|---:|
| US | 1,323,633 | 3.45906 | 5.583% | 5.416% | 17.020% | 71.981% |
| India | 883,188 | 3.46454 | 5.588% | 5.375% | 16.976% | 72.062% |

Country vs 0/1/2/3+ match-count contingency: chi-square=2.8106, df=3, p=0.421752, Cramer V=0.001129. Counts are near-identical in practical terms; noise difficulty differs much more than cardinality. This descriptive test assumes independent S1 and is not a predictive validation score.

## Full positive-pair similarities

All 7,638,365 positives are compared. High similarity means Jaro-Winkler >=0.90; difficult means both name and address JW <0.70. Empty fields are tracked separately. Numeric/postal overlap denominator is **all** positives; both-present rates are provided in the JSON. Acronyms include all normalized tokens; not a semantic acronym detector. Accent-fold equality is not cross-script transliteration.

| Statistic | All | India | US |
|---|---:|---:|---:|
| exact_name_rate | 21.848% | 15.890% | 25.829% |
| exact_address_rate | 8.266% | 7.596% | 8.714% |
| high_name_jw_rate | 68.305% | 53.624% | 78.117% |
| high_address_jw_rate | 41.027% | 32.779% | 46.540% |
| numeric_overlap_rate | 79.878% | 82.592% | 78.065% |
| numeric_both_present_rate | 87.202% | 86.280% | 87.819% |
| postal_overlap_rate | 4.803% | 0.222% | 7.865% |
| postal_both_present_rate | 5.147% | 0.224% | 8.437% |
| acronym_equal_rate | 37.266% | 32.233% | 40.630% |
| accent_fold_equal_rate | 25.787% | 18.753% | 30.487% |
| nonascii_target_name_rate | 13.888% | 23.512% | 7.455% |
| missing_address_rate | 4.412% | 3.925% | 4.738% |
| both_jw_under_07_rate | 2.998% | 6.813% | 0.448% |

S2/S3 also differ: exact-address agreement is materially lower in S3. Keep retrieval-source indicators and inspect source-specific calibration rather than assume identical noise. See the full by-source rates in docs/audit_evidence/pairs.json.

## Difficult positives and deceptive negatives

- Difficult positive examples include Indic-script names paired with Latin reference names; address component removal/reordering; injected leading house numbers; changed state abbreviations; severe aliases or domain-style names with little token overlap.
- Some positives have near-zero name evidence, so name-only blocking needs address/numeric rescue.
- Same-name nonmatches often occur in different cities; a particularly dangerous local example has the same business name and same street/city with house number 531 versus 532. Do not search these records online.
- Prefix/word-overlap similarity must not override contradictory numeric/address evidence blindly. Conversely, injected leading numbers in positives mean a universal number-mismatch veto will also fail.

SHA256 prefix 000..007 S1 sample against ALL training S2/S3, exact normalized name plus country, excluding labeled positives. Not prevalence across all negatives.
Sample S1 count: 4,220.

- US: 33,117 exact-name nonmatches involving 917 sampled S1; high address JW rate=0.00302%; exact address rate=0.00000%.
- India: 5,360 exact-name nonmatches involving 678 sampled S1; high address JW rate=0.00000%; exact address rate=0.00000%.

Across all training data, country+exact-normalized-name blocks contain 21,680,763 pairs. Largest block contributes 78,588 pairs. This block alone has poor positive recall despite sizable fan-out.

## Train/test overlap and leakage

Within each corresponding source, exact IDs and full raw (name,address,country) rows have zero train/test overlap. This does not prove absence of all near-duplicate or cross-source real-world overlap. No test-label copying or identity inference is performed. ID/order artifacts are excluded as model features.

Entity fold manifests cover all S1 and target ownership. Development fold has 441,103 S1; locked fold 4 remains unscored. Training-target view excludes ownership folds 0/4, with a measured zero forbidden-target count. Future training code must use that ownership restriction for both positive and negative pairs. Country-transfer and stricter name-collision grouping are designed but not yet run.

## Diagnostic baseline

EXP-001: exact nonblank normalized name+address+country, full 10,320,219 training targets, development fold only. Macro F0.5=0.08327203; micro precision=100.000%; recall/candidate recall=1.314%; 20,075 scored candidates. Singleton F0.5=1; non-singleton F0.5=0.02894529. All-empty control=0.05594612.

This is a no-training sanity baseline. No high competitive score, hidden test score or final submission is claimed. It demonstrates that conservative exact matching alone cannot solve the recall problem.

## Evidence and reproducibility

- Authoritative full-file statistics: docs/audit_evidence/audit.json; positive and near-neighbor measurements: docs/audit_evidence/pairs.json; script/transliteration measurements: docs/audit_evidence/supplement.json.
- Every TSV SHA256 is in docs/audit_evidence/audit.json; baseline commit/config and metrics in docs/audit_evidence/EXP-001.json.
- Detailed numeric arrays and suffix frequencies are preserved in JSON, including all fields and sources. Source code and run commands are in RUNBOOK.md.
- Historical failed audit logs and earlier normalization outputs remain under outputs/audit. The authoritative normalization preserves Unicode combining marks; earlier AUDIT-003/PAIR-001 are superseded.
- No learned-model training, test prediction, portal submission, paid cloud job, or external record lookup in this phase.
