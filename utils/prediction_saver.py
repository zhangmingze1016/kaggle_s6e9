"""Versioned submissions, OOF bundles and experiment records."""
import fcntl
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


class PredictionSaver:
    def __init__(self, output_dir='predictions', experiment_file='experiments.csv'):
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.experiment_file=Path(experiment_file)
        self.experiment_file.parent.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def _probabilities(values, size):
        values=np.asarray(values,dtype=float)
        if values.shape!=(size,) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError('Predictions must be a finite probability vector matching IDs')
        return values

    def save(self, ids, predictions, score, params, fold_mean=None, fold_std=None,
             best_iterations=None, target='Will_Buy_EV', *, train_ids=None,
             y_true=None, oof_predictions=None, fold_ids=None):
        ids=np.asarray(ids)
        if pd.isna(ids).any() or not pd.Index(ids).is_unique:
            raise ValueError('Test IDs must be present and unique')
        predictions=self._probabilities(predictions,len(ids))
        bundle=None
        provided=[x is not None for x in (train_ids,y_true,oof_predictions,fold_ids)]
        if any(provided):
            if not all(provided):
                raise ValueError('OOF saving requires IDs, labels, predictions and folds')
            train_ids=np.asarray(train_ids)
            if pd.isna(train_ids).any() or not pd.Index(train_ids).is_unique:
                raise ValueError('Training IDs must be present and unique')
            y_true=np.asarray(y_true)
            fold_ids=np.asarray(fold_ids)
            if y_true.shape!=(len(train_ids),) or not np.isin(y_true,[0,1]).all():
                raise ValueError('OOF labels must be aligned binary values')
            if fold_ids.shape!=(len(train_ids),) or not np.isfinite(fold_ids).all() or (fold_ids<0).any():
                raise ValueError('Every training row needs an OOF fold')
            bundle=dict(train_ids=train_ids,y_true=y_true,
                oof_pred=self._probabilities(oof_predictions,len(train_ids)),
                fold_ids=fold_ids,test_ids=ids,test_pred=predictions)
        if not np.isfinite(score):
            raise ValueError('Score must be finite')
        # Serialize allocations/log updates between this project's new runners.
        # Locks are local Unix locks (macOS/Linux).
        with (self.experiment_file.parent/'.prediction_saver.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            numbers=[int(m.group(1)) for p in self.output_dir.glob('prediction_v*.csv')
                     if (m:=re.match(r'prediction_v(\d+)_',p.name))]
            version=f'v{max(numbers,default=0)+1:03d}'
            model=params.get('model','model')
            short={'LightGBM':'lgbm','XGBoost':'xgb','CatBoost':'cb','Ensemble':'blend'}.get(model,model.lower())
            parts=[f'prediction_{version}',short]
            if params.get('feature_engineering'):
                parts+=['fe',params.get('feature_recipe','legacy')]
            if params.get('target_encoding'):
                parts.append('te3')
            if params.get('income_neighbors'):
                parts.append('neighbors')
            for keys,prefix in [(('depth','max_depth'),'d'),(('learning_rate',),'lr'),
                                (('iterations','n_estimators'),'iter')]:
                for key in keys:
                    if key in params:
                        parts.append(f'{prefix}{params[key]}');break
            if params.get('sample_size') is not None:
                parts.append(f'sample{params["sample_size"]}')
            parts.append(f'auc_{score:.5f}')
            filename=self.output_dir/('_'.join(parts)+'.csv')
            pd.DataFrame({'id':ids,target:predictions}).to_csv(filename,index=False)
            if bundle is not None:
                np.savez_compressed(filename.with_suffix('.npz'),**bundle)
            metadata={'version':version,**params,'oof_auc':float(score),
                'fold_mean_auc':fold_mean,'fold_std_auc':fold_std}
            if best_iterations is not None:
                metadata['best_iterations']=','.join(map(str,best_iterations))
                metadata['mean_best_iteration']=float(np.mean(best_iterations))
            filename.with_suffix('.json').write_text(json.dumps(metadata,indent=2,default=str))
            record=pd.DataFrame([metadata])
            if self.experiment_file.exists() and self.experiment_file.stat().st_size:
                record=pd.concat([pd.read_csv(self.experiment_file),record],ignore_index=True)
            temp=self.experiment_file.with_suffix('.tmp.csv')
            record.to_csv(temp,index=False)
            temp.replace(self.experiment_file)
        print(f'Saved {filename}; OOF AUC={score:.7f}',flush=True)
        return filename
