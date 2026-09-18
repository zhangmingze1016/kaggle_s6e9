"""ID-aligned, OOF-guided candidate blends; no leaderboard-label tuning."""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_bundles(paths):
    bundles=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            b={key:data[key] for key in ('train_ids','test_ids','y_true','fold_ids','oof_pred','test_pred')}
        for id_key,pred_key in [('train_ids','oof_pred'),('test_ids','test_pred')]:
            ids=b[id_key];p=b[pred_key]
            if not pd.Index(ids).is_unique or pd.isna(ids).any():
                raise ValueError(f'{path}: duplicate/missing IDs')
            if p.shape!=(len(ids),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError(f'{path}: invalid predictions')
        if b['y_true'].shape!=b['train_ids'].shape or not np.isin(b['y_true'],[0,1]).all():
            raise ValueError(f'{path}: invalid labels')
        if b['fold_ids'].shape!=b['train_ids'].shape or (b['fold_ids']<0).any():
            raise ValueError(f'{path}: invalid folds')
        if bundles:
            ref=bundles[0]
            for id_key,fields in [('train_ids',('y_true','fold_ids','oof_pred')),
                                  ('test_ids',('test_pred',))]:
                if len(b[id_key])!=len(ref[id_key]):
                    raise ValueError('Bundles have different row counts; do not mix samples/full runs')
                index=pd.Index(b[id_key]).get_indexer(ref[id_key])
                if (index<0).any():
                    raise ValueError('Bundle ID sets differ')
                b[id_key]=b[id_key][index]
                for field in fields:
                    b[field]=b[field][index]
            if not np.array_equal(ref['y_true'],b['y_true']):
                raise ValueError('Bundle labels differ')
            if not np.array_equal(ref['fold_ids'],b['fold_ids']):
                raise ValueError('Base models must use identical folds for this comparison')
        bundles.append(b)
    if len(bundles)<2:
        raise ValueError('Provide at least two independently trained OOF bundles')
    return bundles


def representation(matrix, mode):
    if mode=='probability':
        return matrix
    return pd.DataFrame(matrix).rank(method='average',pct=True).to_numpy()


def choose_blend(bundles, seed=20260917, challenger_weights=None):
    """Select on half of OOF rows, report remaining rows without re-tuning.

    The audit is an OOF diagnostic, NOT fully nested validation: base learners
    trained on other folds may have seen labels of rows in the other half.
    """
    y=bundles[0]['y_true']
    oof=np.column_stack([b['oof_pred'] for b in bundles])
    test=np.column_stack([b['test_pred'] for b in bundles])
    selection,audit=train_test_split(np.arange(len(y)),test_size=.5,random_state=seed,stratify=y)
    n=len(bundles)
    candidates=[np.eye(n)[i] for i in range(n)]+[np.full(n,1/n)]
    # Small deterministic grid instead of a high-dimensional leaderboard search.
    for left,right in itertools.combinations(range(n),2):
        for weight in (.25,.5,.75):
            w=np.zeros(n);w[left]=weight;w[right]=1-weight;candidates.append(w)
    if challenger_weights is not None:
        if n != 2 or any(not 0 < w < 1 for w in challenger_weights):
            raise ValueError('Challenger weights require two bundles and weights between 0 and 1')
        candidates = [np.array([1., 0.])] + [np.array([1-w, w]) for w in challenger_weights]
    best=None
    for mode in ('probability','rank'):
        values=representation(oof[selection],mode)
        for weights in candidates:
            score=float(roc_auc_score(y[selection],values@weights))
            if best is None or score>best['selection_auc']+1e-12:
                best={'mode':mode,'weights':weights.tolist(),'selection_auc':score}
    weights=np.asarray(best['weights']);mode=best['mode']
    best['audit_auc']=float(roc_auc_score(y[audit],representation(oof[audit],mode)@weights))
    blended=representation(oof,mode)@weights
    best['full_oof_auc_after_selection']=float(roc_auc_score(y,blended))
    best['single_oof_auc']=[float(roc_auc_score(y,oof[:,i])) for i in range(n)]
    best['single_audit_auc']=[float(roc_auc_score(y[audit],oof[audit,i])) for i in range(n)]
    best['audit_gain_over_baseline']=best['audit_auc']-best['single_audit_auc'][0]
    best['challenger_weights']=challenger_weights
    best['prediction_correlation']=pd.DataFrame(oof).corr().to_numpy().tolist()
    best['audit_note']='OOF diagnostic, not fully nested or independent of base model training.'
    best['selection_seed']=seed
    return blended,representation(test,mode)@weights,best
