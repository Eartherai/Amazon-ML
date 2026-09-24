# Leakage checklist

- [x] Test labels unavailable and not inferred.
- [x] No record sent to external search/services.
- [ ] Train/test identifier and exact-text overlap audited (unlabeled diagnostics only).
- [ ] Ground truth IDs, duplicate IDs, coverage and shared ownership checked.
- [ ] Positive ownership components kept in one fold.
- [ ] Validation targets excluded from training positives AND negatives.
- [ ] Train-fit vocabularies, frequency statistics, augmentations and model features.
- [ ] Development/calibration/locked evaluation roles separate.
- [ ] US->India and India->US robustness evaluated.
- [ ] Shared name/address collision groups used for a stricter stress split.
- [ ] OOF mining and calibration without fold leakage.
- [ ] Candidate recall measured on all relevant positives including zero-candidate entities.
- [ ] Full retrieval pool or clearly reported distractor pool used.
- [ ] No IDs, ID numeric values, row order or split artifacts used as predictors.
- [ ] All predictions from stored final scored candidates; exact empty handling.
- [ ] Public leaderboard not used as a label source.
