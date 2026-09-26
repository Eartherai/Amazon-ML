# Candidate budget and Pareto frontier (CL-027)

Organizer update (2026-09-26): `candidate_pairs.tsv` is part of the final submission, the code that produces it is reviewed, and a smaller candidate set per Source 1 entity ranks higher beyond the leaderboard. Candidates here means the pairs the final (expensive) matcher sees; pruning happens inside the cascade before the matcher, never after it.

All numbers are out-of-fold on labeled train S1 with exact per-S1 truth; retrieval misses count against recall. Oracle F0.5 predicts exactly the true pairs inside the candidate set (empty truth scores 1). Fold4 CLOSED. Scripts: `scripts/classical/candidate_budget.py`, `scripts/classical/candidate_restricted_macro.py`, `scripts/classical/fold3_stack_dump.py`.

## Headline

1. Sparse retrieval, not pruning, is the ceiling. The raw char3 name+address top-100 union (~197/S1) has link recall 0.9666 and oracle 0.9885. Stage-1 LightGBM top-12 keeps almost all of it (0.9652 / 0.9881 on the same 20k), and top-8 still gives oracle 0.9874.
2. Dense bi-encoder retrieval is what lifts the ceiling. Fold-3 sparse top-12 plus dense top-10 new pairs reaches link recall 0.9953 and oracle 0.9987.
3. Cascade with cheap pruning: the stage-1 p>=0.01 band inside top-12, plus dense top-10 pruned by the cheap text-only pair model (p_text>=0.05), gives **6.67 candidates/S1 at oracle 0.9974** (link recall 0.9909).
4. Measured end-to-end macro F0.5 on fold-3 India/US (63,102 S1, stage-2 + e5-base CE stack + ownership + dense rescue). Restricting the matcher to the pruned set loses nothing:

| system | cand/S1 | macro F0.5 |
|---|---:|---:|
| current SUB005 recipe (top-12 + dense top-10) | 16.73 | 0.980591 |
| band p>=0.005 + dense p_text>=0.05 | 7.50 | **0.980835** |
| band p>=0.02 + dense top-10 p_text>=0.5 | **5.60** | 0.980787 |
| band p>=0.05 + dense p_text>=0.5 | 4.70 | 0.980019 |
| band p>=0.1 + dense p_text>=0.5 | 4.20 | 0.978668 |

Pareto frontier (fold-3 India/US): P1 best score is 0.980835 at 7.50/S1; P3 is 0.980787 at 5.60/S1; P4 maximum efficiency is 0.980019 at 4.70/S1. The current 16.7/S1 system is dominated. The 0.995-oracle budget is about 5.9–6.7/S1 (band 0.01–0.02 + pruned dense).

Honesty caveats:
- Dense pairs are still *decided* by the text-only model at p_text>=0.8, the same model used as their pruner. A clean cascade needs the cross-encoder to score the pruned dense survivors; that is the next job.
- Stack context features (stage-2 sibling features, rank, gap) use first-stage scores over the stage-1 top-12, which is cheap stage information.
- France has no labels; its candidate behaviour is unmeasured.

## Full end-to-end restricted-candidate results (fold-3)

| system | cand/S1 | macro F0.5 | India | US | pairs |
|---|---|---|---|---|---|
| band p>=0.1 & rank<=12 + dense top-10 p_text>=0.5 | 4.20 | 0.978668 | 0.97619 | 0.98033 | 265,171 |
| band p>=0.05 & rank<=6 + dense top-5 p_text>=0.5 | 4.35 | 0.978642 | 0.97623 | 0.98026 | 274,709 |
| band p>=0.05 & rank<=12 + dense top-10 p_text>=0.5 | 4.70 | 0.980019 | 0.97756 | 0.98167 | 296,241 |
| band p>=0.05 & rank<=12 + dense top-10 p_text>=0.2 | 4.76 | 0.980019 | 0.97756 | 0.98167 | 300,251 |
| band p>=0.02 & rank<=6 + dense top-10 p_text>=0.2 | 4.93 | 0.979388 | 0.97683 | 0.98110 | 311,373 |
| band p>=0.03 & rank<=12 + dense top-10 p_text>=0.2 | 5.27 | 0.980578 | 0.97806 | 0.98226 | 332,727 |
| band p>=0.02 & rank<=8 + dense top-10 p_text>=0.2 | 5.42 | 0.980700 | 0.97820 | 0.98237 | 341,796 |
| band p>=0.02 & rank<=12 + dense top-3 p_text>=0.2 | 5.55 | 0.979436 | 0.97619 | 0.98161 | 350,471 |
| band p>=0.02 & rank<=12 + dense top-10 p_text>=0.8 | 5.58 | 0.980787 | 0.97835 | 0.98242 | 351,950 |
| band p>=0.02 & rank<=12 + dense top-5 p_text>=0.2 | 5.60 | 0.980705 | 0.97839 | 0.98226 | 353,273 |
| band p>=0.02 & rank<=12 + dense top-10 p_text>=0.5 | 5.60 | 0.980787 | 0.97835 | 0.98242 | 353,380 |
| band p>=0.02 (top-12) + dense top-10 p_text>=0.2 | 5.66 | 0.980787 | 0.97835 | 0.98242 | 357,390 |
| band p>=0.02 (top-12) + dense top-10 p_text>=0.1 | 5.73 | 0.980787 | 0.97835 | 0.98242 | 361,805 |
| band p>=0.02 (top-12) + dense top-10 p_text>=0.05 | 5.88 | 0.980787 | 0.97835 | 0.98242 | 370,777 |
| band p>=0.02 (top-12) + dense top-10 p_text>=0.02 | 6.15 | 0.980787 | 0.97835 | 0.98242 | 387,946 |
| band p>=0.01 & rank<=8 + dense top-10 p_text>=0.05 | 6.17 | 0.980715 | 0.97825 | 0.98237 | 389,013 |
| band p>=0.01 + dense top-5 p_text>=0.05 | 6.40 | 0.980729 | 0.97844 | 0.98226 | 403,801 |
| band p>=0.01 (top-12) + dense top-10 p_text>=0.2 | 6.41 | 0.980811 | 0.97840 | 0.98242 | 404,601 |
| band p>=0.01 & rank<=10 + dense top-10 p_text>=0.05 | 6.48 | 0.980792 | 0.97837 | 0.98241 | 408,814 |
| band p>=0.01 (top-12) + dense top-10 p_text>=0.1 | 6.48 | 0.980811 | 0.97840 | 0.98242 | 409,016 |
| band p>=0.01 (top-12) + dense top-10 p_text>=0.05 | 6.62 | 0.980811 | 0.97840 | 0.98242 | 417,988 |
| band p>=0.01 (top-12) + dense top-10 p_text>=0.02 | 6.90 | 0.980811 | 0.97840 | 0.98242 | 435,157 |
| band p>=0.005 (top-12) + dense top-10 p_text>=0.2 | 7.29 | 0.980835 | 0.97843 | 0.98244 | 459,849 |
| band p>=0.005 (top-12) + dense top-10 p_text>=0.1 | 7.36 | 0.980835 | 0.97843 | 0.98244 | 464,264 |
| band p>=0.005 (top-12) + dense top-10 p_text>=0.05 | 7.50 | 0.980835 | 0.97843 | 0.98244 | 473,236 |
| band p>=0.005 (top-12) + dense top-10 p_text>=0.02 | 7.77 | 0.980835 | 0.97843 | 0.98244 | 490,405 |
| P0 sparse top-12, stack+own (no dense) | 11.98 | 0.973857 | 0.96519 | 0.97966 | 756,051 |
| P1 sparse top-12 + dense top-10 (current SUB005 recipe) | 16.73 | 0.980591 | 0.97829 | 0.98214 | 1,055,629 |

## Raw-pool budget curve (P4-B-001, 20k natural S1, nested stage-1 OOF)

| system | link_recall | complete_entity_recall | oracle_f05 | avg | p50 | p90 | p95 | p99 | max | pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| RAW char3 name+addr top100 union | 0.9666 | 0.9031 | 0.9885 | 197.18 | 197.00 | 200.00 | 200.00 | 200.00 | 200 | 3,943,627 |
| RAW stage1 top-1 | 0.2704 | 0.0513 | 0.6892 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 20,000 |
| RAW stage1 top-2 | 0.5191 | 0.2136 | 0.8630 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2 | 40,000 |
| RAW stage1 top-3 | 0.7137 | 0.4296 | 0.9352 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3 | 60,000 |
| RAW stage1 top-5 | 0.9109 | 0.7654 | 0.9798 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5 | 100,000 |
| RAW stage1 top-8 | 0.9616 | 0.8880 | 0.9874 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8 | 160,000 |
| RAW stage1 top-10 | 0.9644 | 0.8964 | 0.9879 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10 | 200,000 |
| RAW stage1 top-12 | 0.9652 | 0.8988 | 0.9881 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12 | 240,000 |
| RAW stage1 top-15 | 0.9657 | 0.9002 | 0.9882 | 15.00 | 15.00 | 15.00 | 15.00 | 15.00 | 15 | 300,000 |
| RAW stage1 top-20 | 0.9661 | 0.9015 | 0.9883 | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 | 20 | 400,000 |
| RAW stage1 top-30 | 0.9663 | 0.9023 | 0.9884 | 30.00 | 30.00 | 30.00 | 30.00 | 30.00 | 30 | 600,000 |
| RAW stage1 top-50 | 0.9666 | 0.9029 | 0.9885 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 | 50 | 1,000,000 |
| RAW stage1 top-100 | 0.9666 | 0.9031 | 0.9885 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100 | 2,000,000 |
| RAW stage1 p>=0.001 | 0.9660 | 0.9012 | 0.9883 | 13.61 | 11.00 | 23.00 | 29.00 | 48.00 | 105 | 272,168 |
| RAW stage1 top-8 & p>=0.001 | 0.9616 | 0.8879 | 0.9874 | 7.61 | 8.00 | 8.00 | 8.00 | 8.00 | 8 | 152,217 |
| RAW stage1 top-12 & p>=0.001 | 0.9651 | 0.8985 | 0.9881 | 10.06 | 11.00 | 12.00 | 12.00 | 12.00 | 12 | 201,215 |
| RAW stage1 top-20 & p>=0.001 | 0.9658 | 0.9008 | 0.9882 | 12.26 | 11.00 | 20.00 | 20.00 | 20.00 | 20 | 245,179 |
| RAW stage1 p>=0.002 | 0.9655 | 0.8997 | 0.9882 | 10.32 | 9.00 | 17.00 | 21.00 | 35.00 | 85 | 206,341 |
| RAW stage1 top-8 & p>=0.002 | 0.9615 | 0.8876 | 0.9874 | 7.13 | 8.00 | 8.00 | 8.00 | 8.00 | 8 | 142,653 |
| RAW stage1 top-12 & p>=0.002 | 0.9648 | 0.8976 | 0.9880 | 8.74 | 9.00 | 12.00 | 12.00 | 12.00 | 12 | 174,782 |
| RAW stage1 top-20 & p>=0.002 | 0.9654 | 0.8994 | 0.9881 | 9.82 | 9.00 | 17.00 | 20.00 | 20.00 | 20 | 196,376 |
| RAW stage1 p>=0.005 | 0.9645 | 0.8966 | 0.9879 | 7.82 | 7.00 | 13.00 | 15.00 | 24.00 | 67 | 156,301 |
| RAW stage1 top-8 & p>=0.005 | 0.9611 | 0.8866 | 0.9872 | 6.41 | 7.00 | 8.00 | 8.00 | 8.00 | 8 | 128,135 |
| RAW stage1 top-12 & p>=0.005 | 0.9640 | 0.8954 | 0.9878 | 7.27 | 7.00 | 12.00 | 12.00 | 12.00 | 12 | 145,319 |
| RAW stage1 top-20 & p>=0.005 | 0.9644 | 0.8963 | 0.9878 | 7.68 | 7.00 | 13.00 | 15.00 | 20.00 | 20 | 153,501 |
| RAW stage1 p>=0.01 | 0.9632 | 0.8927 | 0.9875 | 6.62 | 6.00 | 10.00 | 13.00 | 19.00 | 52 | 132,361 |
| RAW stage1 top-8 & p>=0.01 | 0.9603 | 0.8839 | 0.9870 | 5.88 | 6.00 | 8.00 | 8.00 | 8.00 | 8 | 117,536 |
| RAW stage1 top-12 & p>=0.01 | 0.9629 | 0.8918 | 0.9874 | 6.39 | 6.00 | 10.00 | 12.00 | 12.00 | 12 | 127,769 |
| RAW stage1 top-20 & p>=0.01 | 0.9631 | 0.8926 | 0.9874 | 6.57 | 6.00 | 10.00 | 13.00 | 19.00 | 20 | 131,485 |
| RAW stage1 p>=0.02 | 0.9608 | 0.8855 | 0.9868 | 5.75 | 5.00 | 9.00 | 11.00 | 15.00 | 39 | 115,017 |
| RAW stage1 top-8 & p>=0.02 | 0.9585 | 0.8786 | 0.9864 | 5.37 | 5.00 | 8.00 | 8.00 | 8.00 | 8 | 107,401 |
| RAW stage1 top-12 & p>=0.02 | 0.9607 | 0.8852 | 0.9867 | 5.67 | 5.00 | 9.00 | 11.00 | 12.00 | 12 | 113,315 |
| RAW stage1 top-20 & p>=0.02 | 0.9608 | 0.8854 | 0.9867 | 5.74 | 5.00 | 9.00 | 11.00 | 15.00 | 20 | 114,785 |
| RAW stage1 p>=0.03 | 0.9591 | 0.8805 | 0.9862 | 5.32 | 5.00 | 8.00 | 10.00 | 13.00 | 34 | 106,314 |
| RAW stage1 top-8 & p>=0.03 | 0.9572 | 0.8744 | 0.9859 | 5.07 | 5.00 | 8.00 | 8.00 | 8.00 | 8 | 101,497 |
| RAW stage1 top-12 & p>=0.03 | 0.9590 | 0.8803 | 0.9862 | 5.27 | 5.00 | 8.00 | 10.00 | 12.00 | 12 | 105,442 |
| RAW stage1 top-20 & p>=0.03 | 0.9591 | 0.8805 | 0.9862 | 5.31 | 5.00 | 8.00 | 10.00 | 13.00 | 20 | 106,216 |
| RAW stage1 p>=0.05 | 0.9556 | 0.8700 | 0.9850 | 4.77 | 5.00 | 7.00 | 8.00 | 11.00 | 28 | 95,435 |
| RAW stage1 top-8 & p>=0.05 | 0.9541 | 0.8654 | 0.9848 | 4.66 | 5.00 | 7.00 | 8.00 | 8.00 | 8 | 93,159 |
| RAW stage1 top-12 & p>=0.05 | 0.9555 | 0.8698 | 0.9850 | 4.75 | 5.00 | 7.00 | 8.00 | 11.00 | 12 | 95,082 |
| RAW stage1 top-20 & p>=0.05 | 0.9556 | 0.8699 | 0.9850 | 4.77 | 5.00 | 7.00 | 8.00 | 11.00 | 20 | 95,404 |
| RAW stage1 p>=0.1 | 0.9485 | 0.8493 | 0.9827 | 4.23 | 4.00 | 7.00 | 8.00 | 9.00 | 26 | 84,678 |
| RAW stage1 top-8 & p>=0.1 | 0.9475 | 0.8462 | 0.9826 | 4.19 | 4.00 | 7.00 | 8.00 | 8.00 | 8 | 83,839 |
| RAW stage1 top-12 & p>=0.1 | 0.9485 | 0.8493 | 0.9827 | 4.23 | 4.00 | 7.00 | 8.00 | 9.00 | 12 | 84,582 |
| RAW stage1 top-20 & p>=0.1 | 0.9485 | 0.8493 | 0.9827 | 4.23 | 4.00 | 7.00 | 8.00 | 9.00 | 20 | 84,670 |
| RAW pruner keep 99.0% of pool positives | 0.9570 | 0.8740 | 0.9856 | 4.94 | 5.00 | 8.00 | 9.00 | 12.00 | 29 | 98,788 |
| RAW pruner keep 99.5% of pool positives | 0.9618 | 0.8884 | 0.9870 | 6.04 | 6.00 | 9.00 | 11.00 | 16.00 | 43 | 120,708 |
| RAW pruner keep 99.7% of pool positives | 0.9637 | 0.8944 | 0.9877 | 6.98 | 6.00 | 11.00 | 13.00 | 20.00 | 62 | 139,622 |
| RAW pruner keep 99.9% of pool positives | 0.9657 | 0.9002 | 0.9882 | 11.33 | 10.00 | 19.00 | 24.00 | 39.00 | 95 | 226,513 |

## Stage-1 top-12 budget curve (200k S1, folds 1-3, OOF base score)

| system | link_recall | complete_entity_recall | oracle_f05 | avg | p50 | p90 | p95 | p99 | max | pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| TOP12 stage1 top-1 | 0.2711 | 0.0531 | 0.6906 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 200,000 |
| TOP12 stage1 top-2 | 0.5205 | 0.2132 | 0.8636 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2 | 400,000 |
| TOP12 stage1 top-3 | 0.7162 | 0.4341 | 0.9357 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3 | 600,000 |
| TOP12 stage1 top-5 | 0.9132 | 0.7694 | 0.9799 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5 | 1,000,000 |
| TOP12 stage1 top-8 | 0.9619 | 0.8885 | 0.9871 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8 | 1,600,000 |
| TOP12 stage1 top-10 | 0.9643 | 0.8958 | 0.9875 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10 | 2,000,000 |
| TOP12 stage1 top-12 | 0.9650 | 0.8980 | 0.9876 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12 | 2,400,000 |
| TOP12 top-12 & p>=0.001 | 0.9650 | 0.8978 | 0.9876 | 9.97 | 11.00 | 12.00 | 12.00 | 12.00 | 12 | 1,994,173 |
| TOP12 top-12 & p>=0.002 | 0.9648 | 0.8971 | 0.9875 | 8.67 | 9.00 | 12.00 | 12.00 | 12.00 | 12 | 1,734,747 |
| TOP12 top-12 & p>=0.005 | 0.9641 | 0.8951 | 0.9873 | 7.21 | 7.00 | 12.00 | 12.00 | 12.00 | 12 | 1,441,335 |
| TOP12 top-12 & p>=0.01 | 0.9631 | 0.8920 | 0.9870 | 6.32 | 6.00 | 10.00 | 12.00 | 12.00 | 12 | 1,264,688 |
| TOP12 top-12 & p>=0.02 | 0.9613 | 0.8864 | 0.9864 | 5.57 | 5.00 | 9.00 | 10.00 | 12.00 | 12 | 1,113,702 |
| TOP12 top-12 & p>=0.03 | 0.9596 | 0.8812 | 0.9858 | 5.17 | 5.00 | 8.00 | 9.00 | 12.00 | 12 | 1,034,685 |
| TOP12 top-12 & p>=0.05 | 0.9564 | 0.8713 | 0.9848 | 4.66 | 4.00 | 7.00 | 8.00 | 11.00 | 12 | 931,406 |
| TOP12 top-12 & p>=0.1 | 0.9498 | 0.8522 | 0.9827 | 4.15 | 4.00 | 7.00 | 7.00 | 9.00 | 12 | 830,503 |

## Fold-3 sparse + dense unions (India/US, 66,666 S1)

| system | link_recall | complete_entity_recall | oracle_f05 | avg | p50 | p90 | p95 | p99 | max | pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| F3 sparse top-12 | 0.9651 | 0.8981 | 0.9876 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12 | 799,992 |
| F3 CE band: top-12 & p>=0.02 | 0.9613 | 0.8864 | 0.9863 | 5.55 | 5.00 | 9.00 | 10.00 | 12.00 | 12 | 369,920 |
| F3 sparse top-12 + dense top-5 new | 0.9882 | 0.9598 | 0.9973 | 12.85 | 13.00 | 14.00 | 15.00 | 16.00 | 17 | 856,487 |
| F3 sparse top-12 + dense top-10 new | 0.9953 | 0.9835 | 0.9987 | 16.71 | 17.00 | 19.00 | 20.00 | 21.00 | 22 | 1,113,787 |
| F3 sparse top-12 + dense top-20 new | 0.9975 | 0.9911 | 0.9993 | 26.16 | 26.00 | 29.00 | 29.00 | 30.00 | 32 | 1,744,125 |
| F3 sparse top-12 + dense top-50 new | 0.9988 | 0.9958 | 0.9996 | 55.61 | 56.00 | 59.00 | 59.00 | 60.00 | 62 | 3,707,491 |
| F3 band p>=0.01 + dense top-10 p_text>=0.05 | 0.9909 | 0.9682 | 0.9974 | 6.67 | 6.00 | 11.00 | 12.00 | 14.00 | 19 | 444,339 |
| F3 band p>=0.01 + dense top-10 p_text>=0.2 | 0.9883 | 0.9598 | 0.9965 | 6.45 | 6.00 | 10.00 | 12.00 | 13.00 | 17 | 430,293 |
| F3 band p>=0.01 + dense top-10 p_text>=0.5 | 0.9854 | 0.9507 | 0.9955 | 6.39 | 6.00 | 10.00 | 12.00 | 12.00 | 17 | 426,073 |
| F3 band p>=0.02 + dense top-10 p_text>=0.05 | 0.9891 | 0.9621 | 0.9968 | 5.91 | 6.00 | 9.00 | 11.00 | 13.00 | 19 | 394,320 |
| F3 band p>=0.02 + dense top-10 p_text>=0.2 | 0.9865 | 0.9538 | 0.9960 | 5.70 | 5.00 | 9.00 | 11.00 | 12.00 | 17 | 380,274 |
| F3 band p>=0.02 + dense top-10 p_text>=0.5 | 0.9836 | 0.9447 | 0.9950 | 5.64 | 5.00 | 9.00 | 10.00 | 12.00 | 17 | 376,054 |
| F3 band p>=0.05 + dense top-10 p_text>=0.05 | 0.9841 | 0.9455 | 0.9953 | 5.00 | 5.00 | 8.00 | 9.00 | 11.00 | 19 | 333,574 |
| F3 band p>=0.05 + dense top-10 p_text>=0.2 | 0.9815 | 0.9374 | 0.9944 | 4.79 | 5.00 | 8.00 | 9.00 | 11.00 | 17 | 319,528 |
| F3 band p>=0.05 + dense top-10 p_text>=0.5 | 0.9786 | 0.9285 | 0.9934 | 4.73 | 5.00 | 7.00 | 8.00 | 11.00 | 17 | 315,308 |

## Throughput and scalability (measured where stated)

- Char3 TF-IDF exact retrieval (P5-INDEX-001, per route per worker): 27–81 queries/s, brute-force sparse dot products per country. It is exact but does not scale to billions as is; production needs an inverted index with IDF-pruned n-gram postings or a MinHash/ANN index, per country partition. That is O(n · postings) instead of O(n · m).
- Stage-1 pruning LightGBM: 13.1M pairs per fold scored and top-12 selected in ~20 s (features precomputed). Linear in pairs.
- Text-only pair model (48 string features + LightGBM): 6,021,465 pairs in 439 s on 8 Mac cores including feature computation (~13.7k pairs/s). Linear in pairs.
- Dense e5-small bi-encoder: exact GPU top-k over each country's targets. Billion scale needs an ANN index (IVF-PQ/HNSW), which is linear-time to encode and sublinear to query.
- Cross-encoder e5-base: runs only on the pruned set (~5.6–7.5/S1 instead of 16.7 or 197), so the reduction versus the raw union is 26–35x and versus the all-pairs space is ~10^6x.
