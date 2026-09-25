"""Cross-fit E5 stacker on the fixed 6k top-12 pairs vs stage-2 v2 alone.

Fold 2 -> eval fold 3 and fold 3 -> eval fold 2. On the training fold, an inner grouped
(by S1) 3-fold OOF picks the decision threshold; then the stacker is refit on the whole
training fold and applied to the eval fold with that threshold. Exact macro per-S1 F0.5
(empty/empty = 1) against FULL truth (targets outside top-12 count as misses).
Variants:
  A  stage-2 v2 prob at the production threshold 0.67 (reference, reproduces 0.963111)
  A2 stage-2 v2 prob, threshold cross-fit with the same inner procedure
  B  stacker without E5  [logit(p2), base_logit, rank, gap_top]
  C  stacker with E5     [logit(p2), e5_logit, base_logit, rank, gap_top]   (decision variant)
  D  diagnostic: C + within-S1 e5 relative features (e5 - max e5, e5 rank)
  E  diagnostic: E5 logit alone, cross-fit threshold
Paired S1 bootstrap (2000 reps) for C-A, C-B, C-A2, D-A.
No test labels, Fold4 untouched (6k set is folds 2/3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path('/Users/earther/Desktop/Amazon ML Challange')
OUT = ROOT / 'outputs/analysis/e5_crossenc_pilot'
EXP044 = Path('/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044')
PARAMS = dict(objective='binary', n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=1.0, verbose=-1, n_jobs=2,
              random_state=20260925, deterministic=True, force_row_wise=True)
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
EPS = 1e-6


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


class Evaluator:
    """Per-S1 F0.5 for a boolean keep-mask over pairs, grouped by query index."""

    def __init__(self, qidx: np.ndarray, t: np.ndarray, truth_sets: list[set]):
        self.qidx, self.t, self.truth = qidx, t, truth_sets
        self.nq = len(truth_sets)

    def per_s1(self, keep: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
        pred = [set() for _ in range(self.nq)]
        for qi, ti in zip(self.qidx[keep], self.t[keep]):
            pred[qi].add(ti)
        qs = range(self.nq) if subset is None else subset
        return np.array([f05(pred[q], self.truth[q]) for q in qs])

    def details(self, keep: np.ndarray, subset: np.ndarray) -> dict:
        tp = fp = fn = 0
        pred = [set() for _ in range(self.nq)]
        for qi, ti in zip(self.qidx[keep], self.t[keep]):
            pred[qi].add(ti)
        for q in subset:
            tp += len(pred[q] & self.truth[q]); fp += len(pred[q] - self.truth[q]); fn += len(self.truth[q] - pred[q])
        return {'pair_precision': tp / max(tp + fp, 1), 'pair_recall': tp / max(tp + fn, 1)}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def best_threshold(ev: Evaluator, prob: np.ndarray, pair_mask: np.ndarray, qsub: np.ndarray) -> float:
    best, best_t = -1.0, 0.5
    for thr in GRID:
        keep = pair_mask & (prob >= thr)
        m = ev.per_s1(keep, qsub).mean()
        if m > best + 1e-12:
            best, best_t = m, float(thr)
    return best_t


def main() -> None:
    z = np.load(ROOT / 'outputs/experiments/CL-003/eval6k-top12-v2.npz', allow_pickle=False)
    q = z['q'].astype(str); t = z['t'].astype(str); p2 = z['p'].astype(np.float64); X = z['X']
    meta = pl.read_ndjson(EXP044 / 'pairs.jsonl').select('q', 't', 'fold', 'country', 'label')
    if not ((meta['q'].to_numpy() == q).all() and (meta['t'].to_numpy() == t).all()):
        raise ValueError('pairs/npz misaligned')
    e5_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / 'job_output/e5_logits_6k.tsv'
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / 'stack_e5_6k.json'
    e5 = pl.read_csv(e5_path, separator='\t', quote_char=None,
                     schema_overrides={'i': pl.Int64, 'q': pl.Utf8, 't': pl.Utf8, 'e5_logit': pl.Float64})
    if not ((e5['i'].to_numpy() == np.arange(len(q))).all() and (e5['q'].to_numpy() == q).all()
            and (e5['t'].to_numpy() == t).all()):
        raise ValueError('E5 logits misaligned')
    e5l = e5['e5_logit'].to_numpy()
    truth = json.loads((EXP044 / 'truth.json').read_text())
    uq, qidx = np.unique(q, return_inverse=True)
    truth_sets = [set(truth[s]) for s in uq]
    ev = Evaluator(qidx, t, truth_sets)
    label = np.array([ti in truth_sets[qi] for qi, ti in zip(qidx, t)], dtype=np.int8)
    if not (label == meta['label'].to_numpy()).all():
        raise ValueError('label mismatch vs EXP-044 pairs')
    fold_pair = meta['fold'].to_numpy()
    q_fold = np.zeros(len(uq), dtype=int); q_fold[qidx] = fold_pair
    q_country = np.empty(len(uq), dtype=object); q_country[qidx] = meta['country'].to_numpy()
    # within-S1 E5 relative features (diagnostic D)
    e5max = np.full(len(uq), -np.inf); np.maximum.at(e5max, qidx, e5l)
    order = np.lexsort((-e5l, qidx)); e5rank = np.empty(len(q), dtype=np.float64)
    pos = np.arange(len(q)); starts = np.r_[0, np.flatnonzero(np.diff(qidx[order])) + 1]
    grp_start = np.repeat(starts, np.diff(np.r_[starts, len(q)]))
    e5rank[order] = pos - grp_start + 1
    base_logit, rank, gap_top = X[:, 1].astype(np.float64), X[:, 2].astype(np.float64), X[:, 3].astype(np.float64)
    feats = {
        'B': np.column_stack([logit(p2), base_logit, rank, gap_top]),
        'C': np.column_stack([logit(p2), e5l, base_logit, rank, gap_top]),
        'D': np.column_stack([logit(p2), e5l, base_logit, rank, gap_top, e5l - e5max[qidx], e5rank]),
    }
    # final (out-of-fold on the 6k) probabilities per variant
    final = {k: np.full(len(q), np.nan) for k in ['B', 'C', 'D']}
    thresholds = {k: {} for k in ['A2', 'B', 'C', 'D', 'E']}
    keep = {'A': p2 >= 0.67}
    keep_parts = {k: np.zeros(len(q), dtype=bool) for k in ['A2', 'B', 'C', 'D', 'E']}
    for train_fold, eval_fold in ((2, 3), (3, 2)):
        tr = fold_pair == train_fold; te = fold_pair == eval_fold
        tr_q = np.flatnonzero(q_fold == train_fold)
        # deterministic grouped inner 3-fold on training S1
        rng = np.random.default_rng(20260925 + train_fold)
        inner = np.empty(len(uq), dtype=int); inner[tr_q] = rng.permutation(len(tr_q)) % 3
        for k, F in feats.items():
            oof = np.full(len(q), np.nan)
            for j in range(3):
                fit = tr & (inner[qidx] != j); hold = tr & (inner[qidx] == j)
                m = lgb.LGBMClassifier(**PARAMS).fit(F[fit], label[fit])
                oof[hold] = m.predict_proba(F[hold])[:, 1]
            thr = best_threshold(ev, oof, tr, tr_q)
            thresholds[k][f'train{train_fold}'] = thr
            m = lgb.LGBMClassifier(**PARAMS).fit(F[tr], label[tr])
            final[k][te] = m.predict_proba(F[te])[:, 1]
            keep_parts[k] |= te & (final[k] >= thr)
        thr = best_threshold(ev, p2, tr, tr_q); thresholds['A2'][f'train{train_fold}'] = thr
        keep_parts['A2'] |= te & (p2 >= thr)
        # E5 alone: sigmoid of logit, threshold chosen on training fold (no model fit needed)
        e5p = 1 / (1 + np.exp(-e5l))
        thr = best_threshold(ev, e5p, tr, tr_q); thresholds['E'][f'train{train_fold}'] = thr
        keep_parts['E'] |= te & (e5p >= thr)
    keep.update(keep_parts)
    allq = np.arange(len(uq))
    per = {k: ev.per_s1(v, allq) for k, v in keep.items()}
    res = {}
    for k, v in per.items():
        res[k] = {'macro_f0_5': float(v.mean()),
                  **{c: float(v[q_country == c].mean()) for c in ('India', 'US')},
                  **{f'fold{f}': float(v[q_fold == f].mean()) for f in (2, 3)},
                  **ev.details(keep[k], allq)}
    rng = np.random.default_rng(7)
    boots = rng.integers(0, len(uq), size=(2000, len(uq)))
    deltas = {}
    for a, b in (('C', 'A'), ('C', 'B'), ('C', 'A2'), ('B', 'A'), ('D', 'A'), ('D', 'B'), ('E', 'A')):
        d = per[a] - per[b]
        bs = d[boots].mean(axis=1)
        deltas[f'{a}-{b}'] = {'delta': float(d.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              'n_s1_changed': int((d != 0).sum()), 'n_better': int((d > 0).sum()), 'n_worse': int((d < 0).sum())}
    # pair-level diagnostics
    from sklearn.metrics import roc_auc_score
    diag = {'auc_e5_logit': float(roc_auc_score(label, e5l)), 'auc_stage2_p': float(roc_auc_score(label, p2)),
            'auc_base': float(roc_auc_score(label, X[:, 0])),
            'spearman_e5_vs_stage2': float(pl.DataFrame({'a': e5l, 'b': p2}).select(pl.corr('a', 'b', method='spearman')).item()),
            'auc_C_oof': float(roc_auc_score(label, final['C'])), 'auc_B_oof': float(roc_auc_score(label, final['B'])),
            'n_pairs': int(len(q)), 'n_pos_in_top12': int(label.sum()),
            'n_truth_targets': int(sum(len(s) for s in truth_sets))}
    delta = deltas['C-A']
    decision = ('PROMOTE' if delta['delta'] >= 0.002 and delta['ci95'][0] > 0 else 'KILL')
    report = {'scope': 'fixed 6k top-12 (72,000 pairs), folds 2/3, cross-fit 2->3 and 3->2, Fold4 CLOSED',
              'e5_checkpoint': 'P5-E5-18K-001 (multilingual-e5-small fine-tuned on 18k fold-1 S1, owner-safe)',
              'params': PARAMS, 'threshold_grid': [float(GRID[0]), float(GRID[-1]), 0.01],
              'thresholds': thresholds, 'results': res, 'paired_bootstrap_2000': deltas,
              'pair_diagnostics': diag, 'decision_rule': 'promote iff C-A delta >= +0.002 and CI low > 0',
              'decision': decision}
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
