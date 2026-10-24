"""Development-only robustness utilities.

No held-out test labels or predictions are used for checkpoint, fusion, or
threshold selection.
"""
from functools import lru_cache
import numpy as np
from sklearn.metrics import f1_score

@lru_cache(maxsize=50000)
def condition_for_path(path):
    import numpy as _np
    try:
        with _np.load(path, allow_pickle=True) as d:
            return str(d["condition"].item()) if "condition" in d.files else "clean"
    except Exception:
        return "clean"

def macro_condition_f1(paths, labels, scores, threshold=0.5):
    labels=np.asarray(labels,dtype=int); scores=np.asarray(scores,dtype=float)
    conditions=np.asarray([condition_for_path(str(p)) for p in paths],dtype=object)
    vals=[]
    for cond in sorted(set(map(str, conditions))):
        m=np.asarray([str(c)==cond for c in conditions],dtype=bool)
        if m.sum():
            vals.append(f1_score(labels[m], (scores[m]>=threshold).astype(int), zero_division=0))
    return float(np.mean(vals)) if vals else 0.0

def condition_balanced_weights(conditions):
    conditions=np.asarray(conditions,dtype=object)
    unique, counts=np.unique(conditions.astype(str), return_counts=True)
    inv={u: len(conditions)/(len(unique)*c) for u,c in zip(unique,counts)}
    w=np.asarray([inv[str(c)] for c in conditions],dtype=np.float64)
    return w / max(float(np.mean(w)), 1e-12)

def choose_condition_balanced_threshold(labels, probs, conditions):
    labels=np.asarray(labels,dtype=int); probs=np.asarray(probs,dtype=float)
    conditions=np.asarray(conditions,dtype=object)
    best_t, best_macro, best_overall = 0.5, -1.0, -1.0
    for t in np.linspace(0.05,0.95,181):
        preds=(probs>=t).astype(int)
        per=[]
        for cond in sorted(set(map(str,conditions))):
            m=np.asarray([str(c)==cond for c in conditions],dtype=bool)
            if m.sum(): per.append(f1_score(labels[m],preds[m],zero_division=0))
        macro=float(np.mean(per)) if per else 0.0
        overall=float(f1_score(labels,preds,zero_division=0))
        if macro > best_macro + 1e-12 or (abs(macro-best_macro)<=1e-12 and overall>best_overall):
            best_t,best_macro,best_overall=float(t),macro,overall
    return best_t,best_macro,best_overall
