"""Exact competition macro F0.5 and explicitly named diagnostics."""
from collections.abc import Mapping, Iterable

def entity_f05(truth: set[str], predicted: set[str]) -> float:
    """Empty/empty earns 1; all other zero-TP cases earn 0."""
    if not truth and not predicted:
        return 1.0
    tp=len(truth & predicted)
    return 1.25*tp/(0.25*len(truth)+len(predicted))

def evaluate(truth: Mapping[str,set[str]], predictions: Mapping[str,set[str]]) -> dict:
    """Require exact S1 coverage so missing rows cannot silently improve scores."""
    if not truth: raise ValueError('Evaluation requires at least one S1 entity')
    if truth.keys()!=predictions.keys(): raise ValueError('Prediction S1 coverage differs from truth')
    scores=[]; single=[]; non_single=[]; tp=fp=fn=0
    for key,true in truth.items():
        pred=predictions[key]; s=entity_f05(true,pred); scores.append(s)
        (non_single if true else single).append(s)
        tp+=len(true & pred); fp+=len(pred-true); fn+=len(true-pred)
    return {'macro_f0_5':sum(scores)/len(scores),
            'micro_precision':tp/(tp+fp) if tp+fp else None,
            'micro_recall':tp/(tp+fn) if tp+fn else None,
            'singleton_f0_5':sum(single)/len(single) if single else None,
            'non_singleton_f0_5':sum(non_single)/len(non_single) if non_single else None,
            'entities':len(truth),'tp':tp,'fp':fp,'fn':fn}

def f05_from_counts(tp:int, predicted:int, actual:int) -> float:
    """Count form independently usable by streaming/SQL evaluations."""
    if min(tp,predicted,actual)<0 or tp>min(predicted,actual): raise ValueError('Invalid counts')
    return 1.0 if predicted==actual==0 else 1.25*tp/(predicted+0.25*actual)
