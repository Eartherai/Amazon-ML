# Loss ledger toward 0.99 (CL-049)

## Calibration points

| quantity | value | source |
|---|---:|---|
| PUBLIC (SUB011, v4 B: QNORM stage-2 + CE stack + CE-decided dense rescue, VSAFE France) | **0.97205** | leaderboard (user) |
| leader | 0.990556 | leaderboard (user) |
| gap to leader | 0.018506 | |
| LOCAL_CORE (v2q + CE stack, 3 folds, 194k S1, no dense) | 0.97529 | CL-042 |
| LOCAL_FULL_F3 (core + dense rescue, fold 3 only, India/US) | 0.98435 | CL-041 |
| PUBLIC - CORE | -0.0032 | |
| PUBLIC - FULL_F3 | -0.0123 | |

The transfer simulator (US<->India, ~0.92) is far below the public score. It is a relative stress test only, not an estimate of public.

The public test also contains France (15% of S1, no labels). If India/US score publicly at the local full level (0.9843) with India/US 85% of S1, then France would be ~0.90. If the dense rescue gain were not real on other folds, India/US would sit near the core (0.9753) and France near ~0.95. The dense-rescue multi-fold confirmation (running) decides between these.

## Update: dense rescue confirmed on all folds (CL-050)

Pooled folds 1-3 (193,289 S1): core 0.975523 -> core + dense 0.984301 (+0.00878; per fold +0.00890 / +0.00854 / +0.00889; India 0.9667 -> 0.9844, US 0.9815 -> 0.9842). The local full system is a genuine ~0.9843. The public gap is therefore not dense optimism. SUB011 used VSAFE France rows (old stage-2 + rescue, no CE stack and no CE dense rescue). With India/US at 0.9843 on 85% of S1, SUB011's 0.97205 implies France ~0.90. **France is the main public lever: each +0.01 of France F0.5 is +0.0015 public.** Next: the full pipeline on France (SUB012 / SUB013).

## Fold-3 loss decomposition of the full primary (63,102 India/US S1, exact macro F0.5)

| component | loss (macro F0.5) | maximum recovery | best experiment |
|---|---:|---:|---|
| total | 0.0157 | | |
| retrieval misses (true pairs outside all candidates) | 0.0031 | 0.0031 (oracle 0.9969) | wider dense for unseen country (France top-50, running); more routes |
| in-candidate missed matches (FN) | **0.0077** | 0.0077 | entity-level / set-completion model (codex3); recall-leaning decisions for S1 already matched; semantic judge (codex1/2) |
| in-candidate false positives (FP) | 0.0048 | 0.0048 | semantic judge on hard pairs (codex1/2); hard-negative CE refit |
| of which singleton S1 given a match | 0.0008 (49 S1) | 0.0008 | singleton predictor (codex3) |
| partially matched S1 (some in-candidate FN) | 4,707 S1 (7.5%) | inside the FN row | set completion |
| S1 with truth but zero TP | 170 S1 | inside FN/retrieval | |

A perfect decision on every hard pair of the test half gives 0.99708 (+0.0125), which is the in-candidate ceiling.

Per country (fold 3): India 0.9847 (oracle 0.99645), US 0.98407 (oracle 0.99716).

## Public gap (not visible locally)

- Dense-rescue optimism: unknown until folds 1/2 are measured (P5-BIENC-E5S-F1/F2-001 running).
- France: unmeasured. Evidence so far: CE-versus-stage-2 agreement on France matches US (0.9837 vs 0.9861); dense retrieval loses 0.0066 of oracle on an unseen country, of which top-50 recovers about 55%.
- Test India/US shift: SUB011's public score is only 0.0032 below the multi-fold core, so any residual shift is small.

## Priorities

1. Multi-fold dense rescue confirmation.
2. Set-level completion for partially matched S1 (largest in-candidate loss).
3. France wide dense (top-50) plus pruning plus CE.
4. Semantic judges for FP/FN on hard pairs, only if the net test delta is positive.
