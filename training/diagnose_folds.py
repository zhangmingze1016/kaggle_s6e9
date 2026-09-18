"""Standalone fold audit; optional bounded raw-feature CV probe, never changes training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def assignments(y, seed, n_splits=5):
    result = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        result[valid] = fold
    return result


def run(args):
    out = Path(args.output)
    if out.resolve().is_relative_to(Path('predictions').resolve()):
        raise ValueError('Diagnostics must be outside predictions/')
    out.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    if not train.id.is_unique or not train.Will_Buy_EV.isin(['Yes','No']).all():
        raise ValueError('Invalid IDs or target')
    y = train.Will_Buy_EV.eq('Yes').astype(int).to_numpy()
    X = train.drop(columns=['id','Will_Buy_EV'])
    numeric = X.select_dtypes(include='number').columns.tolist()
    cats = [c for c in X if c not in numeric]
    allfolds = {seed: assignments(y, seed) for seed in args.seeds}
    primary = assignments(y,42)
    summary, numbers, categories, missing, contrasts = [], [], [], [], []
    structural = X.copy()
    structural['id'] = train.id
    structural['row_position'] = np.arange(len(train))
    hashes = pd.util.hash_pandas_object(X,index=False)
    duplicate = hashes.duplicated(keep=False)
    hashframe = pd.DataFrame({'hash':hashes,'y':y,'fold':primary})
    groups = hashframe.groupby('hash').agg(labels=('y','nunique'),folds=('fold','nunique'))
    testhash = pd.util.hash_pandas_object(test[X.columns],index=False)
    overlap = hashes.isin(testhash)
    idbins = pd.qcut(train.id,10,duplicates='drop').astype(str)
    for seed, folds in allfolds.items():
        for fold in range(5):
            mask = folds == fold
            summary.append(dict(seed=seed,fold=fold+1,rows=int(mask.sum()),positive_rate=float(y[mask].mean()),
                id_min=int(train.id[mask].min()),id_max=int(train.id[mask].max()),
                duplicate_rate=float(duplicate[mask].mean()),test_exact_overlap_rate=float(overlap[mask].mean())))
            for col in structural.select_dtypes(include='number'):
                series = structural.loc[mask,col]
                record = dict(seed=seed,fold=fold+1,feature=col,mean=series.mean(),std=series.std())
                record.update({f'q{int(q*100):02d}':series.quantile(q) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]})
                numbers.append(record)
            for col in X:
                missing.append(dict(seed=seed,fold=fold+1,feature=col,missing_rate=float(X.loc[mask,col].isna().mean())))
            for col in cats+['id_decile']:
                values = idbins if col=='id_decile' else X[col].astype('string').fillna('__MISSING__')
                tab = pd.DataFrame({'value':values[mask].to_numpy(),'target':y[mask]}).groupby('value').target.agg(['size','mean'])
                for value,row in tab.iterrows():
                    categories.append(dict(seed=seed,fold=fold+1,feature=col,value=value,count=int(row['size']),fraction=row['size']/mask.sum(),positive_rate=row['mean']))
    for label in ['all',0,1]:
        keep = np.ones(len(y),bool) if label=='all' else y==label
        for col in structural.select_dtypes(include='number'):
            a=structural.loc[(primary==0)&keep,col].dropna(); b=structural.loc[(primary==2)&keep,col].dropna()
            scale=np.sqrt((a.var()+b.var())/2)
            contrasts.append(dict(target=str(label),feature=col,fold1_mean=a.mean(),fold3_mean=b.mean(),
                standardized_mean_difference=(a.mean()-b.mean())/scale if scale else 0.,ks_statistic=ks_2samp(a,b).statistic))
    for name, data in [('fold_summary',summary),('numeric',numbers),('categorical',categories),('missing',missing),('fold1_vs_fold3',contrasts)]:
        pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    # Category shifts in Fold 1 vs Fold 3, including conditional label rates.
    catframe=pd.DataFrame(categories)
    tv=[]
    for feature,group in catframe[catframe.seed==42].groupby('feature'):
        table=group.pivot(index='value',columns='fold',values='fraction').fillna(0)
        tv.append(dict(feature=feature,total_variation=float((table[1]-table[3]).abs().sum()/2)))
    pd.DataFrame(tv).to_csv(out/'categorical_fold1_vs_fold3.csv',index=False)
    cohorts=[]
    for name,values in [('id_decile',idbins),('id_mod10',train.id%10),
                         ('income_repeated',X.Annual_Income_USD.duplicated(keep=False)),
                         ('category_combination',X[cats].astype(str).agg('|'.join,axis=1))]:
        tab=pd.DataFrame({'cohort':values,'target':y,'fold':primary+1}).groupby(['cohort','fold']).target.agg(['size','mean']).reset_index()
        tab.insert(0,'kind',name);cohorts.append(tab)
    pd.concat(cohorts,ignore_index=True).to_csv(out/'structural_cohorts.csv',index=False)
    audit=[]
    for path in sorted(Path('artifacts').rglob('config.json')):
        cfg=json.loads(path.read_text())
        if not {'n_splits','random_seed','sample_size'} <= cfg.keys(): continue
        if cfg['sample_size'] is not None:
            audit.append(dict(path=str(path),status='sample run; not compared'));continue
        expected=assignments(y,cfg['random_seed'],cfg['n_splits'])
        checks=[]
        for fold in range(cfg['n_splits']):
            checkpoint=path.parent/f'fold_{fold+1:02d}.npz'
            if checkpoint.exists():
                with np.load(checkpoint,allow_pickle=False) as d:
                    checks.append(bool(np.array_equal(d['valid_indices'],np.flatnonzero(expected==fold))))
        audit.append(dict(path=str(path),seed=cfg['random_seed'],n_splits=cfg['n_splits'],available_folds=len(checks),
                          matches_declared_split=all(checks) if checks else None,
                          same_primary_partition=bool(np.array_equal(expected,primary))))
    bundle_scores=[]; bundle_audit=[]; score_profiles=[]
    for path in sorted(Path('artifacts/predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:
            idx=pd.Index(d['train_ids']).get_indexer(train.id)
            if len(d['train_ids'])!=len(train) or (idx<0).any():
                bundle_audit.append(dict(path=str(path),matches_primary=False,reason='different IDs'));continue
            equal=bool(np.array_equal(d['fold_ids'][idx],primary) and np.array_equal(d['y_true'][idx],y))
            bundle_audit.append(dict(path=str(path),matches_primary=equal))
            if equal:
                for fold in range(5):
                    mask=primary==fold
                    values=d['oof_pred'][idx][mask]; labels=y[mask]
                    bundle_scores.append(dict(version=path.name.split('_')[1],fold=fold+1,auc=roc_auc_score(labels,values)))
                    if path.name.startswith('prediction_v027_'):
                        positive=np.sort(values[labels==1]);negative=np.sort(values[labels==0])
                        placements_p=(np.searchsorted(negative,positive,'left')+np.searchsorted(negative,positive,'right'))/(2*len(negative))
                        placements_n=1-(np.searchsorted(positive,negative,'left')+np.searchsorted(positive,negative,'right'))/(2*len(positive))
                        se=np.sqrt(placements_p.var(ddof=1)/len(positive)+placements_n.var(ddof=1)/len(negative))
                        score_profiles.append(dict(fold=fold+1,auc=roc_auc_score(labels,values),conditional_auc_se=se,
                            positive_mean=positive.mean(),negative_mean=negative.mean(),positive_q10=np.quantile(positive,.1),negative_q90=np.quantile(negative,.9)))
    pd.DataFrame(score_profiles).to_csv(out/'v027_score_profiles.csv',index=False)
    scores=pd.DataFrame(bundle_scores)
    if not scores.empty:
        pivot=scores.pivot(index='fold',columns='version',values='auc')
        pivot.to_csv(out/'model_fold_auc.csv')
        if 'v027' in pivot: pivot.subtract(pivot.v027,axis=0).to_csv(out/'model_fold_delta_vs_v027.csv')
        if 'v020' in pivot: pivot.subtract(pivot.v020,axis=0).to_csv(out/'model_fold_delta_vs_v020.csv')
    facts=dict(configuration=vars(args),rows=len(train),positive_rate=float(y.mean()),id_monotonic=bool(train.id.is_monotonic_increasing),
        id_equals_row_position=bool(np.array_equal(train.id,np.arange(len(train)))),duplicate_rows=int(duplicate.sum()),
        duplicate_groups=int((hashes.value_counts()>1).sum()),conflicting_label_groups=int((groups.labels>1).sum()),
        groups_spanning_folds=int((groups.folds>1).sum()),train_test_exact_feature_matches=int(overlap.sum()),
        original_data_note='No original-source dataset supplied: source matches and generator mechanism cannot be established.',
        target_lag1_correlation=float(pd.Series(y).autocorr()),
        deterministic=bool(np.array_equal(primary,assignments(y,42))),
        fingerprints={str(s):hashlib.sha256(f.tobytes()).hexdigest() for s,f in allfolds.items()},
        checkpoint_audit=audit,bundle_audit=bundle_audit)
    (out/'audit.json').write_text(json.dumps(facts,indent=2))
    print(json.dumps({k:v for k,v in facts.items() if k not in ['checkpoint_audit','bundle_audit','fingerprints']},indent=2),flush=True)
    if args.probe:
        from lightgbm import LGBMClassifier
        probe=X.copy()
        for c in cats: probe[c]=probe[c].astype('category')
        (out/'probe_config.json').write_text(json.dumps(dict(seeds=args.seeds,iterations=args.probe_iterations,model_seed=42,learning_rate=.08,num_leaves=15,max_depth=4,min_child_samples=50,features='raw; ID excluded; no target encoding; no early stopping'),indent=2))
        records=[]
        # Fixed complexity, raw features, no early stopping/TE or seed selection.
        for seed,folds in allfolds.items():
            for fold in range(5):
                valid=folds==fold
                model=LGBMClassifier(n_estimators=args.probe_iterations,learning_rate=.08,num_leaves=15,
                    max_depth=4,min_child_samples=50,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                model.fit(probe.loc[~valid],y[~valid])
                auc=roc_auc_score(y[valid],model.predict_proba(probe.loc[valid])[:,1])
                records.append(dict(seed=seed,fold=fold+1,auc=float(auc)))
                pd.DataFrame(records).to_csv(out/'alternate_seed_probe.csv',index=False)
                print(f'Probe seed={seed} fold={fold+1} AUC={auc:.7f}',flush=True)
    print(f'Reports: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/train.csv');p.add_argument('--test',default='data/test.csv')
    p.add_argument('--output',default='artifacts/fold_diagnostics')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026,3407,8888])
    p.add_argument('--probe',action='store_true',help='Train 25 bounded raw-feature models; no main pipeline changes')
    p.add_argument('--probe-iterations',type=int,default=150)
    args=p.parse_args()
    if args.probe_iterations<1:p.error('probe-iterations must be positive')
    run(args)


if __name__=='__main__':main()
