"""Shared CV mechanics; public training entry points remain model-specific."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils.data_loader import EVDataLoader
from utils.feature_engineering import FoldFeatureEngineer, TripleTargetEncoder
from utils.prediction_saver import PredictionSaver


DEFAULTS = {
    'LightGBM': dict(n_estimators=100000, learning_rate=.005, max_depth=7,
        num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=0,
        colsample_bytree=1., reg_alpha=.1, reg_lambda=2., max_bin=255,
        early_stopping_rounds=500, is_unbalance=True, n_splits=10,
        feature_engineering=True, feature_recipe='notebook', target_encoding=True),
    'XGBoost': dict(n_estimators=3000, learning_rate=.05, max_depth=6,
        min_child_weight=1., subsample=.8, colsample_bytree=.8,
        reg_alpha=0., reg_lambda=1., max_bin=256, early_stopping_rounds=200,
        n_splits=5, feature_engineering=False, feature_recipe='legacy', target_encoding=False),
    'CatBoost': dict(iterations=3000, learning_rate=.05, depth=4, l2_leaf_reg=3.,
        rsm=1., early_stopping_rounds=200, n_splits=5,
        feature_engineering=False, feature_recipe='legacy', target_encoding=False),
}
STRONG = {
    'LightGBM': dict(n_estimators=3500, learning_rate=.02, max_depth=5,
        num_leaves=32, min_child_samples=10, subsample=.8, subsample_freq=1,
        colsample_bytree=.3, reg_alpha=.071, reg_lambda=2., max_bin=255,
        early_stopping_rounds=150, is_unbalance=False),
    'XGBoost': dict(n_estimators=2400, learning_rate=.03, max_depth=6,
        min_child_weight=12., subsample=.82, colsample_bytree=.55,
        reg_alpha=.08, reg_lambda=3., max_bin=512, early_stopping_rounds=150),
    'CatBoost': dict(iterations=3500, learning_rate=.05, depth=6,
        l2_leaf_reg=5., rsm=.8, early_stopping_rounds=200),
}


def parse_args(model, argv=None):
    parser = argparse.ArgumentParser(description=f'{model}: reproducible CV and OOF artifacts')
    parser.add_argument('--preset', choices=['baseline', 'strong'], default='baseline')
    preliminary, _ = parser.parse_known_args(argv)
    defaults = DEFAULTS[model].copy()
    if preliminary.preset == 'strong':
        defaults.update(STRONG[model])
        defaults.update(n_splits=5, feature_engineering=True,
                        feature_recipe='multiscale', target_encoding=True)
    for key, value in defaults.items():
        if key in {'feature_recipe'}:
            parser.add_argument('--feature-recipe', choices=['legacy', 'notebook', 'multiscale'], default=value)
        elif isinstance(value, bool):
            parser.add_argument('--'+key.replace('_','-'), action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument('--'+key.replace('_','-'), type=type(value), default=value)
    parser.add_argument('--te-scope', choices=['numeric', 'all', 'bins'], default='numeric')
    parser.add_argument('--income-neighbors', action=argparse.BooleanOptionalAction, default=preliminary.preset=='strong')
    parser.add_argument('--local-windows', action='store_true', help='Independent local-window feature view instead of exact income/commute TE and income neighbors')
    parser.add_argument('--te-cv', type=int, default=5)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu','gpu','cuda'], default='cpu')
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--log-period', type=int, default=100)
    parser.add_argument('--train-path', default='data/train.csv')
    parser.add_argument('--test-path', default='data/test.csv')
    parser.add_argument('--original-data', nargs='?', const='data/external/ev_adoption.csv', default=None,
                        help='Opt in to 13 external-only target-statistic features; optional source CSV path')
    parser.add_argument('--output-dir', default='predictions')
    parser.add_argument('--artifact-dir', default='artifacts/predictions')
    parser.add_argument('--experiment-file', default='artifacts/experiments.csv')
    parser.add_argument('--run-root', default='artifacts/runs')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sample-size', type=int, default=None, help='Stratified development sample; not full-data CV')
    args = parser.parse_args(argv)
    if args.local_windows and (not args.feature_engineering or args.feature_recipe != 'multiscale' or not args.target_encoding or args.original_data):
        parser.error('local-windows requires multiscale target encoding without original-data')
    if args.original_data and (not args.feature_engineering or args.feature_recipe != 'multiscale'):
        parser.error('original-data currently requires feature engineering with the multiscale recipe')
    if args.n_splits < 2 or args.te_cv < 2:
        parser.error('n-splits and te-cv must be at least 2')
    if args.sample_size is not None and args.sample_size < 20:
        parser.error('sample-size must be at least 20')
    if args.log_period < 0 or args.early_stopping_rounds < 1:
        parser.error('log-period must be >= 0 and early-stopping-rounds >= 1')
    return args


def make_model(name, a):
    if name == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{k:getattr(a,k) for k in (
            'n_estimators','learning_rate','max_depth','num_leaves','min_child_samples',
            'subsample','subsample_freq','colsample_bytree','reg_alpha','reg_lambda',
            'max_bin','is_unbalance')}, objective='binary', metric='auc',
            device=a.device, random_state=a.model_seed, n_jobs=a.n_jobs, verbosity=-1)
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(**{k:getattr(a,k) for k in (
            'n_estimators','learning_rate','max_depth','min_child_weight','subsample',
            'colsample_bytree','reg_alpha','reg_lambda','max_bin','early_stopping_rounds')},
            objective='binary:logistic',eval_metric='auc',tree_method='hist',
            device='cpu' if a.device=='cpu' else 'cuda',
            random_state=a.model_seed,n_jobs=a.n_jobs)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**{k:getattr(a,k) for k in (
        'iterations','learning_rate','depth','l2_leaf_reg','rsm')},
        loss_function='Logloss',eval_metric='AUC',random_seed=a.model_seed,
        task_type='CPU' if a.device=='cpu' else 'GPU',thread_count=a.n_jobs,
        allow_writing_files=False,verbose=False)


def prepare_categories(frames, name, cat_cols):
    frames = [f.copy() for f in frames]
    for col in cat_cols:
        values = [f[col].astype('string').fillna('__MISSING__').astype(str) for f in frames]
        if name == 'CatBoost':
            for f, v in zip(frames, values):
                f[col] = v
        else:
            # Vocabulary learned from the training fold; unseen levels are missing.
            dtype = pd.CategoricalDtype(sorted(values[0].unique()))
            for f, v in zip(frames, values):
                f[col] = v.where(v.isin(dtype.categories)).astype(dtype)
    if name == 'XGBoost' and cat_cols:
        frames = [pd.get_dummies(f, columns=cat_cols, dtype=np.int8) for f in frames]
    return frames


def fit_predict(name, a, frames, y_fit, y_valid, cat_cols):
    fit, valid, test = prepare_categories(frames, name, cat_cols)
    model = make_model(name, a)
    if name == 'LightGBM':
        import lightgbm as lgb
        model.fit(fit, y_fit, eval_X=valid, eval_y=y_valid,
            categorical_feature=cat_cols, callbacks=[
                lgb.early_stopping(a.early_stopping_rounds,first_metric_only=True),
                lgb.log_evaluation(a.log_period)])
        best = int(model.best_iteration_)
    elif name == 'XGBoost':
        model.fit(fit,y_fit,eval_set=[(valid,y_valid)],verbose=a.log_period or False)
        best = int(model.best_iteration)+1
    else:
        model.fit(fit,y_fit,cat_features=cat_cols,eval_set=(valid,y_valid),
            early_stopping_rounds=a.early_stopping_rounds,verbose=a.log_period or False)
        best = int(model.get_best_iteration())+1
    return model.predict_proba(valid)[:,1], model.predict_proba(test)[:,1], best


def file_hash(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run_cv(name, args):
    params = {'model':name, **vars(args)}
    if args.local_windows:
        params['effective_income_neighbors'] = False
    versions = {p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn',name.lower())}
    identity = {k:v for k,v in params.items() if k not in (
        'resume','output_dir','artifact_dir','experiment_file','run_root','log_period')}
    identity['data_hashes'] = {p:file_hash(p) for p in (args.train_path,args.test_path)}
    identity['code_hashes'] = {p:file_hash(p) for p in (
        'utils/cv_runner.py','utils/data_loader.py','utils/feature_engineering.py')}
    if args.original_data:
        identity['data_hashes'][args.original_data] = file_hash(args.original_data)
        identity['code_hashes']['utils/original_data.py'] = file_hash('utils/original_data.py')
    if args.local_windows:
        identity['code_hashes']['utils/local_windows.py'] = file_hash('utils/local_windows.py')
    identity['versions'] = versions
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    run_dir = Path(args.run_root)/f'{name.lower()}_{signature}'
    run_dir.mkdir(parents=True,exist_ok=True)
    params['run_signature'] = signature
    print(json.dumps(params,indent=2),flush=True)
    print(f'Run directory: {run_dir}',flush=True)
    (run_dir/'config.json').write_text(json.dumps(identity,indent=2))
    loader = EVDataLoader(args.train_path,args.test_path,args.feature_engineering,
        args.feature_recipe,args.sample_size,args.random_seed)
    X,y,T,test_ids,cats = loader.load()
    y = y.eq('Yes').astype(int)
    print(f'Train {X.shape}; test {T.shape}; categorical={len(cats)}',flush=True)
    if y.value_counts().min() < args.n_splits:
        raise ValueError('Not enough observations per class for outer CV')
    original = None
    if args.original_data:
        from utils.original_data import OriginalTargetStatistics, load_source
        original = OriginalTargetStatistics().fit(load_source(args.original_data),
            [pd.read_csv(args.train_path), pd.read_csv(args.test_path)])
        (run_dir/'original_data_audit.json').write_text(json.dumps(original.audit_,indent=2))
        print(f'External source: {original.audit_}',flush=True)
    oof = np.full(len(X),np.nan)
    test_pred = np.zeros(len(T))
    folds = np.full(len(X),-1,dtype=np.int16)
    scores, best_iterations = [], []
    cv = StratifiedKFold(args.n_splits,shuffle=True,random_state=args.random_seed)
    for fold,(fit,valid) in enumerate(cv.split(X,y)):
        started=time.monotonic()
        checkpoint=run_dir/f'fold_{fold+1:02d}.npz'
        if args.resume and checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as data:
                if not np.array_equal(data['valid_indices'],valid):
                    raise ValueError('Checkpoint split mismatch')
                pv,pt,best=data['valid_pred'],data['test_pred'],int(data['best_iteration'])
            print(f'Fold {fold+1}: resumed',flush=True)
        else:
            print(f'Fold {fold+1}/{args.n_splits}: train={len(fit)}, valid={len(valid)}',flush=True)
            frames=[X.iloc[fit],X.iloc[valid],T]
            if args.local_windows:
                from utils.local_windows import LocalWindowEncoder, replace_exact_target_features
                local = LocalWindowEncoder(args.te_cv,args.random_seed)
                windows = [local.fit_transform(frames[0],y.iloc[fit]),
                           local.transform(frames[1]),local.transform(frames[2])]
            if args.target_encoding or args.income_neighbors or (
                args.feature_engineering and args.feature_recipe=='multiscale'):
                if (args.feature_recipe=='notebook' and args.te_scope=='numeric'
                        and args.target_encoding and not args.income_neighbors):
                    encoder=TripleTargetEncoder(args.te_cv,args.random_seed)
                else:
                    encoder=FoldFeatureEngineer(
                        args.feature_recipe if args.feature_engineering else 'raw',
                        args.target_encoding,args.te_scope,args.te_cv,args.random_seed,
                        args.income_neighbors and not args.local_windows)
                frames=[encoder.fit_transform(frames[0],y.iloc[fit]),
                        encoder.transform(frames[1]),encoder.transform(frames[2])]
            if args.local_windows:
                frames = [replace_exact_target_features(frame,window) for frame,window in zip(frames,windows)]
            if original is not None:
                frames = [pd.concat([frame, original.transform(frame)],axis=1) for frame in frames]
            pv,pt,best=fit_predict(name,args,frames,y.iloc[fit],y.iloc[valid],cats)
            del frames
            temp=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temp,valid_indices=valid,valid_pred=pv,
                test_pred=pt,best_iteration=best)
            temp.replace(checkpoint)
        if len(pv)!=len(valid) or len(pt)!=len(T) or not np.isfinite(pv).all() or not np.isfinite(pt).all():
            raise ValueError('Incomplete/nonfinite fold predictions')
        oof[valid]=pv
        folds[valid]=fold
        test_pred+=pt/args.n_splits
        score=float(roc_auc_score(y.iloc[valid],pv))
        scores.append(score);best_iterations.append(best)
        print(f'Fold {fold+1} AUC={score:.7f}; trees={best}; seconds={time.monotonic()-started:.1f}',flush=True)
        (run_dir/'progress.json').write_text(json.dumps({'completed_folds':fold+1,
            'total_folds':args.n_splits,'fold_scores':scores,'best_iterations':best_iterations},indent=2))
    score=float(roc_auc_score(y,oof))
    params.update(sample_rows=len(X),test_rows=len(T),run_dir=str(run_dir))
    saver=PredictionSaver(args.output_dir,args.experiment_file,args.artifact_dir)
    path=saver.save(
        test_ids,test_pred,score,params,float(np.mean(scores)),float(np.std(scores)),
        best_iterations,train_ids=loader.train_ids,y_true=y,oof_predictions=oof,fold_ids=folds)
    result={'oof_auc':score,'fold_scores':scores,'submission':str(path),
            'bundle':str(saver.artifact_path(path,'.npz')),'params':params}
    (run_dir/'result.json').write_text(json.dumps(result,indent=2))
    print(f'Complete OOF AUC={score:.7f}; {path}',flush=True)
    return result
