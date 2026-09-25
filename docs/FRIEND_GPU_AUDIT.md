# Teammate GPU research audit — 2026-09-25

Source: `/Users/earther/Downloads/amazon-ml-challenge-2026-main/` as supplied by the user. This is a code-and-results snapshot, not a model/artifact snapshot: there are no notebooks, CE-002 model weights, scored pair Parquet files, or full submission outputs in the supplied directory. The teammate's exact checkpoint cannot be reproduced or deployed directly from it. Do not conflate its metrics with the current EXP-044/045 OOF population or public leaderboard.

## Verified useful evidence

- `results/BASE-OOF-001/metrics.json` reports a 20,000-S1 51-feature baseline macro F0.5 of **0.9323416557**, aligned with our older 20k 0.9318965293 on a related sample/protocol.
- `results/HYB/summary_20k.json` reports **0.9626469469** for a LightGBM with fine-tuned E5 cross-encoder score as an extra feature, and **0.9622351034** with deployment-style CE gate `baseline OOF score >= 0.02`; corresponding base in that snapshot is 0.932342. The score increase is substantial but not an independent reproduction here.
- `results/HYB2/summary_20k.json` reports **0.9652662252** with target-centric exclusive assignment, LightGBM, cross-encoder, and bi-encoder cosine. Its hidden-competitor stress result is 0.9647066267. Plain pairwise decision with the same model is 0.9592254230.
- `results/TC-LEX-20K/summary.json` reports lexical target-centric assignment **0.9366891264** versus same-model pair threshold **0.9303424214**, both on 20k; the original 51-feature base is 0.932342. This supports testing global ownership with proper competitors, but does not imply +0.006 over our stronger current model.
- Source audit: `src/dl/crossencoder.py` trains on a separate training pool and explicitly filters targets owned by evaluation/validation S1 and Fold4. `src/prep.py` derives that owner table from training labels and reconstructs the sample split. This is an apparently leakage-conscious design; complete independent artifact/data hash verification is impossible from this source-only snapshot.
- The teammate uses `intfloat/multilingual-e5-small` at pinned revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; the [official model card](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md) declares MIT. The model card is a license/source check, not a validation of the teammate's scores.

## Action for this branch

1. Preserve SUB-002 classical inference and organizer validation. The user's first upload remains 0.913 public; three portal uploads remain by their report.
2. Promote the independently measured EXP-045 fine-tuned MiniLM reranker on the current 51-feature model: 6,000 S1 held-fold crossfit macro **0.9474834591 vs 0.9325591312** base, **+0.0149243278**; India +0.0256132661; micro precision 0.9832791→0.9854716. Only `score >= 0.4 or top2` sidecar candidates were used. France stays classical in the first neural variant because it lacks labels.
3. Test a limited G5 prototype, then full reranking only if it meets throughput and $100 total budget gates. Keep this neural TSV separate, run the organizer validator in default and strict ID modes, and let the user choose portal upload.
4. Teammate's feature-stacking and target-centric ideas are the next high-value OOF tests if model artifacts or time become available. Rebuilding the full CE-002 pipeline from source would require retraining and a fresh fold-safe OOF comparison; do not treat its reported 0.965 as our local score.

No external identity lookup or business enrichment was used in this audit. Fold4 remains closed.
