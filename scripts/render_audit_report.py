"""Create a human-readable audit from measured local JSON outputs."""
from pathlib import Path
import json,math
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import chi2_contingency
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'outputs/audit/AUDIT-004/audit.json').read_text())
p=json.loads((root/'outputs/audit/PAIR-002/pairs.json').read_text())
s=json.loads((root/'outputs/audit/supplement-001.json').read_text())
m=json.loads((root/'outputs/audit/memory-001.json').read_text())
v=json.loads((root/'artifacts/validation/v1/manifest.json').read_text())
b=json.loads((root/'outputs/experiments/EXP-001/metrics.json').read_text())
lines=['# Complete dataset audit','',f"Generated {datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()}.",'',
'All six source TSVs and the full ground truth were parsed. Exact aggregates cover every row. Near-duplicate probes and transliteration samples have explicitly limited scopes. No test labels were accessed or inferred. Official files were not modified.','','## Core findings','',
'- 24,229,173 source records; 2,206,821 train S1 entities and 1,732,544 test S1 entities.',
'- 7,638,365 positive links. Every ground-truth S1 and target ID exists; no duplicate S1 labels, repeated label IDs, or shared target owners.',
'- All positive links have equal country labels. Country remains an arbitrary string; France is present only in test.',
'- Raw S1 training names are ASCII. Many target variants use Indic scripts or diacritics. India requires more than Latin lexical matching.',
'- Do not deduplicate output IDs by text: S2/S3 contain repeated record text but each record ID remains a legitimate distinct target.',
'- Strict name AND address equality has excellent development precision but retrieves only 1.31% of positive links. It is not competitive.','','## Inventory and memory','',
'Raw textual columns are all VARCHAR strings; no numeric coercion, default pandas NA coercion, or country enumeration. Blank cells remain empty strings. Null counts and textual NA markers are separate. Disk sizes use MiB (2^20 bytes). The database compressed raw/label data to about 1.51 GiB before derived tables; the full audit process peaked near 3.24 GiB RSS on the Mac. These are actual process/database measurements, not estimated pandas memory. Later derived tables and probes increase artifact disk use. No 24M-row Python object dataframe was constructed.','',
'| File | Rows | Disk MiB | Unique IDs | Unique raw names | Unique raw addresses | Raw duplicate excess | Normalized duplicate excess |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in a['files'].items():
 lines.append(f"| {name} | {r['n']:,} | {r['file_bytes']/2**20:.2f} | {r['unique_ids']:,} | {r['unique_names']:,} | {r['unique_addresses']:,} | {r['exact_duplicates']['redundant_rows']:,} | {r['normalized_duplicates']['redundant_rows']:,} |")
lines+=['',f"Ground truth: {a['ground_truth']['integrity']['record_count']:,} rows, {a['ground_truth']['file_bytes']/2**20:.2f} MiB. Source IDs are unique and prefixes agree with source files.",'',
'## Missingness and empties','',
'Each entry is count (percentage). `null_count` is parser nulls; `empty` is zero-length text; `blank` includes whitespace-only values; markers are literal NA/N/A/null/none/nan strings. Literal markers are not automatically replaced.','',
'| File | Field | Null | Empty | Blank after trim | Textual markers |','|---|---|---:|---:|---:|---:|']
for name,r in a['files'].items():
 for field,f in r['fields'].items():
  cells=[f"{f[k]:,} ({f[k]/r['n']:.3%})" for k in ['null_count','empty_strings','blank_after_trim','textual_missing_markers']]
  lines.append('| '+name+' | '+field+' | '+' | '.join(cells)+' |')
lines+=['','## Country distribution','', '| File | Country | Rows | Share |','|---|---|---:|---:|']
for name,r in a['files'].items():
 for x in r['countries']:lines.append(f"| {name} | {x['country']} | {x['n']:,} | {x['n']/r['n']:.3%} |")
lines+=['','Test S1 shifts from about 60% US/40% India to 38.3% US/46.8% India/15.0% France. Train-only overall validation therefore overweights US relative to test. No France quality estimate is available from labels.','','## Field lengths','',
'Unicode codepoint lengths; not byte lengths or tokenizer counts. Quantiles are exact across each full source file.','','| File | Field | Mean | Min | p1 | p25 | Median | p75 | p95 | p99 | Max |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,r in a['files'].items():
 for field in ['business_name','business_address']:
  f=r['fields'][field];vals=[f['mean_length'],f['min_length'],*f['length_quantiles'],f['max_length']]
  lines.append('| '+name+' | '+field+' | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
lines+=['','## Characters, scripts and transliteration','',
'No language detector is assumed: Unicode block coverage is a script proxy, and Latin text cannot distinguish language or romanized Indic content. Counts can overlap when records mix scripts. Name and address non-ASCII counts are raw-field diagnostics.','',
'| File | Non-ASCII names | Non-ASCII addresses | Names with digits | Addresses with digits |','|---|---:|---:|---:|---:|']
for name,r in a['files'].items():
 x=r['text_patterns'];lines.append(f"| {name} | {x['nonascii_name']:,} | {x['nonascii_address']:,} | {x['name_with_digits']:,} | {x['address_with_digits']:,} |")
for name,x in s['script_coverage'].items():
 lines+=['',f"**{name} script coverage:** "+', '.join(f'{k}={n:,}' for k,n in x.items())+'.']
probe=s['transliteration_probe']
lines+=['',probe['scope'],'',f"Engine: {probe['engine']}. Runtime for transformation/scoring: {probe['runtime_seconds']:.2f}s.",'',
'| Country | Sample pairs | Raw/light JW >= .90 | Transliterated JW >= .90 | Raw JW < .70 rescued to >= .90 | Exact after transliteration |','|---|---:|---:|---:|---:|---:|']
for name,x in probe['by_country'].items():
 lines.append(f"| {name} | {x['pairs']:,} | {x['original_jw_ge_09']:.2%} | {x['transliterated_jw_ge_09']:.2%} | {x['original_low_transliterated_high']:.2%} | {x['transliterated_exact']:.2%} |")
lines+=['','This is positive-similarity evidence, not measured retrieval recall or precision. Next test must include realistic distractors and transliteration collisions. Foundation is a local generic transform, not external enrichment; Linux ICU output parity remains to be tested.','','## Legal endings and numeric/postal shapes','',
'The authoritative ending counts use a token boundary and a documented generic suffix list. These are lexical proxies, not verified legal forms. Token-boundary counts in memory.json supersede the initial suffix-string probes. Do not use these counts to strip legal identity information automatically. Numeric token counts use digit runs; postals are only five/six-digit shapes, not real geocoded codes.','']
for name,r in a['files'].items():
 suffix=[f"{x['suffix']}={x['n']:,}" for x in m['files'][name]['legal_suffix_token_counts'] if x['suffix']]
 nums=', '.join(f"{x['numeric_tokens']}:{x['n']:,}" for x in r['numeric_token_counts'])
 x=r['text_patterns']
 lines += [f"**{name}**",'', '- Ending counts (token boundary): '+', '.join(suffix)+'.','- Address numeric-token distribution (tokens:records): '+nums+'.',f"- Postal5-like: {x['postal5_like']:,}; postal6-like: {x['postal6_like']:,}.",'']
lines+=['## Exact and near duplicates','',
'Raw duplicate key=(name,address,country). Normalized key uses NFC+lower and Unicode letters, marks and numbers. Excess is rows minus distinct keys; ID uniqueness remains separate. S1 train has zero exact or normalized full-text duplicates. S1 test has one normalized duplicate group (two IDs): both must still be output. Raw names alone repeat frequently.','',p['near_duplicate_scope'],'',
'| File | Adjacent pairs evaluated | Near pairs |','|---|---:|---:|']
for name,x in p['near_duplicate_probes'].items():lines.append(f"| {name} | {x['compared_pairs']:,} | {x['near_pairs']:,} |")
lines+=['','No claim of exhaustive fuzzy duplicate discovery: that would require a much broader blocking study. This is a deterministic lower-bound diagnostic over every source.','','## Ground-truth topology','',
'| Matches per S1 | Entities | Share |','|---|---:|---:|']
N=a['ground_truth']['integrity']['record_count']
for x in a['ground_truth']['match_counts']:lines.append(f"| {x['n_matches']} | {x['n']:,} | {x['n']/N:.3%} |")
counts={x['n_matches']:x['n'] for x in a['ground_truth']['match_counts']};ge3=sum(n for k,n in counts.items() if k>=3)
lines+=['',f"Grouped: 0={counts[0]/N:.3%}; 1={counts[1]/N:.3%}; 2={counts[2]/N:.3%}; 3+={ge3/N:.3%}.",'',
'| Source relationship | Entities | Share of all S1 |','|---|---:|---:|']
for x in a['ground_truth']['source_distribution']:
 name='both' if x['has_s2'] and x['has_s3'] else 'S2 only' if x['has_s2'] else 'S3 only' if x['has_s3'] else 'no matches'
 lines.append(f"| {name} | {x['n']:,} | {x['n']/N:.3%} |")
lines+=['','All 7,638,365 labeled targets have exactly one S1 owner. This supports **target-only** exclusivity experiments; S1 has many targets so one-to-one assignment is inappropriate. Training targets without any labeled owner number '+f"{v['unowned_targets']:,}"+'. They are distractors under the supplied exhaustive-label interpretation.','','## Country differences','',
'| Country | S1 | Mean matches | Singletons | One | Two | Three+ |','|---|---:|---:|---:|---:|---:|---:|']
cont=[]
for x in a['ground_truth']['by_country']:
 cont.append([x['singletons'],x['one_match'],x['two_matches'],x['three_plus']])
 lines.append(f"| {x['country']} | {x['n']:,} | {x['mean_matches']:.5f} | {x['singletons']/x['n']:.3%} | {x['one_match']/x['n']:.3%} | {x['two_matches']/x['n']:.3%} | {x['three_plus']/x['n']:.3%} |")
chi,pval,dof,_=chi2_contingency(cont);cram=math.sqrt(chi/N)
lines+=['',f"Country vs 0/1/2/3+ match-count contingency: chi-square={chi:.4f}, df={dof}, p={pval:.6g}, Cramer V={cram:.6f}. Counts are near-identical in practical terms; noise difficulty differs much more than cardinality. This descriptive test assumes independent S1 and is not a predictive validation score.",'',
'## Full positive-pair similarities','',
'All 7,638,365 positives are compared. High similarity means Jaro-Winkler >=0.90; difficult means both name and address JW <0.70. Empty fields are tracked separately. Numeric/postal overlap denominator is **all** positives; both-present rates are provided in the JSON. Acronyms include all normalized tokens; not a semantic acronym detector. Accent-fold equality is not cross-script transliteration.','',
'| Statistic | All | India | US |','|---|---:|---:|---:|']
by={x['country']:x for x in p['positive_by_country']}
for key in ['exact_name_rate','exact_address_rate','high_name_jw_rate','high_address_jw_rate','numeric_overlap_rate','numeric_both_present_rate','postal_overlap_rate','postal_both_present_rate','acronym_equal_rate','accent_fold_equal_rate','nonascii_target_name_rate','missing_address_rate','both_jw_under_07_rate']:
 lines.append(f"| {key} | {p['positive_all'][key]:.3%} | {by['India'][key]:.3%} | {by['US'][key]:.3%} |")
lines+=['','S2/S3 also differ: exact-address agreement is materially lower in S3. Keep retrieval-source indicators and inspect source-specific calibration rather than assume identical noise. See the full by-source rates in docs/audit_evidence/pairs.json.','','## Difficult positives and deceptive negatives','',
'- Difficult positive examples include Indic-script names paired with Latin reference names; address component removal/reordering; injected leading house numbers; changed state abbreviations; severe aliases or domain-style names with little token overlap.',
'- Some positives have near-zero name evidence, so name-only blocking needs address/numeric rescue.',
'- Same-name nonmatches often occur in different cities; a particularly dangerous local example has the same business name and same street/city with house number 531 versus 532. Do not search these records online.',
'- Prefix/word-overlap similarity must not override contradictory numeric/address evidence blindly. Conversely, injected leading numbers in positives mean a universal number-mismatch veto will also fail.','',
p['hard_negative_probe']['scope'],f"Sample S1 count: {p['hard_negative_probe']['sample_s1']:,}.",'']
for x in p['hard_negative_probe']['statistics']:
 lines.append(f"- {x['country']}: {x['pairs']:,} exact-name nonmatches involving {x['affected_s1']:,} sampled S1; high address JW rate={x['high_address_jw_rate']:.5%}; exact address rate={x['exact_address_rate']:.5%}.")
lines+=['',f"Across all training data, country+exact-normalized-name blocks contain {p['exact_name_block_volume']['pairs']:,} pairs. Largest block contributes {p['exact_name_block_volume']['largest_block_pairs']:,} pairs. This block alone has poor positive recall despite sizable fan-out.",'',
'## Train/test overlap and leakage','',
'Within each corresponding source, exact IDs and full raw (name,address,country) rows have zero train/test overlap. This does not prove absence of all near-duplicate or cross-source real-world overlap. No test-label copying or identity inference is performed. ID/order artifacts are excluded as model features.',
'', 'Entity fold manifests cover all S1 and target ownership. Development fold has 441,103 S1; locked fold 4 remains unscored. Training-target view excludes ownership folds 0/4, with a measured zero forbidden-target count. Future training code must use that ownership restriction for both positive and negative pairs. Country-transfer and stricter name-collision grouping are designed but not yet run.','','## Diagnostic baseline','',
f"EXP-001: exact nonblank normalized name+address+country, full 10,320,219 training targets, development fold only. Macro F0.5={b['overall']['macro_f0_5']:.8f}; micro precision={b['overall']['micro_precision']:.3%}; recall/candidate recall={b['overall']['micro_recall']:.3%}; {b['overall']['candidate_pairs']:,} scored candidates. Singleton F0.5=1; non-singleton F0.5={b['overall']['non_singleton_f0_5']:.8f}. All-empty control={b['all_empty_control']['macro_f0_5']:.8f}.",
'', 'This is a no-training sanity baseline. No high competitive score, hidden test score or final submission is claimed. It demonstrates that conservative exact matching alone cannot solve the recall problem.','','## Evidence and reproducibility','',
'- Authoritative full-file statistics: docs/audit_evidence/audit.json; positive and near-neighbor measurements: docs/audit_evidence/pairs.json; script/transliteration measurements: docs/audit_evidence/supplement.json.',
'- Every TSV SHA256 is in docs/audit_evidence/audit.json; baseline commit/config and metrics in docs/audit_evidence/EXP-001.json.',
'- Detailed numeric arrays and suffix frequencies are preserved in JSON, including all fields and sources. Source code and run commands are in RUNBOOK.md.',
'- Historical failed audit logs and earlier normalization outputs remain under outputs/audit. The authoritative normalization preserves Unicode combining marks; earlier AUDIT-003/PAIR-001 are superseded.',
'- No learned-model training, test prediction, portal submission, paid cloud job, or external record lookup in this phase.']
memory_lines=['## Measured materialized memory','',m['scope'],'','| Source | Polars visible buffers MiB |','|---|---:|']
for name,x in m['files'].items():
    assert x['rows']==a['files'][name]['n']
    memory_lines.append(f"| {name} | {x['polars_estimated_buffer_bytes']/2**20:.2f} |")
memory_lines+=['',f"Sequential memory-audit peak process RSS: {m['peak_rss_gib']:.3f} GiB, elapsed {m['elapsed_seconds']:.2f}s. Raw source content was read independently by both DuckDB and Polars and row counts agreed.",'']
idx=lines.index('## Missingness and empties');lines[idx:idx]=memory_lines
text='\n'.join(lines)+'\n'
text=text.replace('near 3.24 GiB',f"near {a['peak_rss_gib']:.2f} GiB")
(root/'DATA_AUDIT.md').write_text(text)
