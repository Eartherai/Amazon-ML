# Phase4 retrieval engine evidence

5k reference retrieval completed in738.51seconds, peak process RSS1.927GiB. It searched all10,320,219 training targets for both routes. Recall96.6503%, complete-positive-entity coverage90.3464%, oracle macro0.989149,197.189 candidates/S1. This is a natural-prevalence development sample, with IDF fitting excluded from OOF1–3.

The previous Python path densifies sparse products and selects topK in Python. [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) fuses sparse multiplication and topN selection. Version1.2.0 has an [Apache2.0 license](https://github.com/ing-bank/sparse_dot_topn/blob/v1.2.0/LICENSE). This is an algorithmic dependency, not a pretrained model or external identity source.

A bounded unlabeled benchmark of1,000 India queries against25,000 targets measured name speedup2.59x and address3.42x, with zero differing candidate sets. This does not establish a full-pool speedup. Name required6 and address2 boundary-tie fallbacks. The integrated kernel additionally falls back for near-ties within1e-6, using the reference calculation and lexical-ID tie break. An identical-vector synthetic test verifies deterministic boundary handling.

The20k run uses this fused kernel with2 threads; the5k reference artifacts remain immutable. Because the natural samples are nested and IDF is identical, compare all5k nested candidate sets across the runs before trusting kernel equivalence at scale. Do not explain a candidate difference as model improvement.

The subset's CSR sizes extrapolate to approximately1.90GiB for names and4.48GiB for addresses before IDs and allocator overhead. These are estimates from one country/subset, not measured full-index sizes. Persisting both locally must respect the8GiB free-space reserve. Current Parquet candidates/features and saved IDF are reusable; full target CSR indexes are not yet persisted.
