import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.cv_runner import parse_args, prepare_categories
from utils.feature_engineering import MultiScaleFeatureEngineer, FoldFeatureEngineer, IncomeNeighborhoodEncoder
from utils.prediction_saver import PredictionSaver
from utils.ensemble import load_bundles, choose_blend


class PipelineTest(unittest.TestCase):
    def test_presets_and_overrides(self):
        a=parse_args('LightGBM',['--preset','strong','--learning-rate','.07'])
        self.assertEqual(a.learning_rate,.07)
        self.assertEqual(a.feature_recipe,'multiscale')
        self.assertTrue(a.income_neighbors)
        self.assertFalse(parse_args('CatBoost',[]).feature_engineering)

    def test_neighborhood_crossfitting(self):
        X=pd.DataFrame({'Annual_Income_USD':np.full(100,50000.)})
        y=np.tile([0,1],50)
        encoder=IncomeNeighborhoodEncoder(resolutions=(32,))
        values=encoder.fit_transform(X,y)
        np.testing.assert_allclose(values.filter(regex='mean$'),.5)
        unknown=encoder.transform(pd.DataFrame({'Annual_Income_USD':[np.nan,1e8]}))
        self.assertTrue(np.isfinite(unknown).all().all())
        np.testing.assert_allclose(unknown.filter(regex='mean$'),.5)

    def test_fold_stats_do_not_learn_test(self):
        raw=pd.read_csv('data/train.csv',nrows=100).drop(columns=['id','Will_Buy_EV'])
        X=MultiScaleFeatureEngineer.transform(raw)
        y=np.tile([0,1],50)
        encoder=FoldFeatureEngineer(recipe='multiscale',income_neighbors=True)
        train=encoder.fit_transform(X,y)
        held=X.iloc[:10].copy();held['Annual_Income_USD']=1e12
        transformed=encoder.transform(held)
        self.assertTrue(train.columns.equals(transformed.columns))
        self.assertTrue((transformed['freq_Annual_Income_USD']==0).all())
        before=encoder.transform(X)
        encoder.transform(held)
        assert_frame_equal(before,encoder.transform(X))

    def test_expanded_bin_te_preserves_baseline_features(self):
        raw=pd.read_csv('data/train.csv',nrows=100).drop(columns=['id','Will_Buy_EV'])
        X=MultiScaleFeatureEngineer.transform(raw)
        y=np.tile([0,1],50)
        base=FoldFeatureEngineer(recipe='multiscale')
        expanded=FoldFeatureEngineer(recipe='multiscale',te_scope='bins')
        a=base.fit_transform(X,y)
        b=expanded.fit_transform(X,y)
        assert_frame_equal(a,b[a.columns])
        extra=set(b)-set(a)
        self.assertEqual(len(extra),24)
        self.assertTrue(all(c.startswith('TE_') for c in extra))
        held=X.iloc[:8].copy()
        held['inc_q50']=1e12
        transformed=expanded.transform(held)
        self.assertTrue(b.columns.equals(transformed.columns))
        self.assertTrue(np.isfinite(transformed[list(extra)]).all().all())
        # Unique income bins: inner validation rows cannot use their own labels.
        unique=X.copy()
        unique['inc_q50']=np.arange(len(X))
        crossfit=FoldFeatureEngineer(recipe='multiscale',te_scope='bins').fit_transform(unique,y)
        np.testing.assert_allclose(crossfit['TE_inc_q50_auto'],.5)

    def test_category_unknown_and_alignment(self):
        train=pd.DataFrame({'cat':['b','a']})
        valid=pd.DataFrame({'cat':['z']})
        a,b,c=prepare_categories([train,valid,valid],'XGBoost',['cat'])
        self.assertTrue(a.columns.equals(b.columns))
        self.assertEqual(b.sum().sum(),0)

    def test_bundle_alignment_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            y=np.tile([0,1],20)
            fold=np.repeat([0,1],20)
            saver=PredictionSaver(p/'predictions',p/'experiments.csv',p/'artifacts')
            paths=[]
            for reverse in (False,True):
                idx=np.arange(40)[::-1] if reverse else np.arange(40)
                tids=np.array([101,100]) if reverse else np.array([100,101])
                preds=np.array([.8,.2]) if reverse else np.array([.2,.8])
                out=saver.save(tids,preds,1.,{'model':'LightGBM'},train_ids=idx,
                    y_true=y[idx],oof_predictions=.1+.8*y[idx],fold_ids=fold[idx])
                paths.append(saver.artifact_path(out,'.npz'))
                self.assertTrue(saver.artifact_path(out,'.json').is_file())
                self.assertTrue(all(f.suffix=='.csv' for f in (p/'predictions').iterdir()))
            bundles=load_bundles(paths)
            np.testing.assert_array_equal(bundles[0]['oof_pred'],bundles[1]['oof_pred'])
            np.testing.assert_array_equal(bundles[0]['test_pred'],bundles[1]['test_pred'])
            oof,test,report=choose_blend(bundles)
            _,_,small=choose_blend(bundles,challenger_weights=[.1,.2,.3])
            self.assertEqual(small['weights'],[1.,0.])
            self.assertEqual(small['audit_gain_over_baseline'],0.)
            with self.assertRaises(ValueError):
                choose_blend(bundles,challenger_weights=[1.1])
            self.assertEqual(report['audit_auc'],1.)
            self.assertAlmostEqual(sum(report['weights']),1.)
            bundles[1]['fold_ids']=1-bundles[1]['fold_ids']
            np.savez_compressed(paths[1],**bundles[1])
            with self.assertRaisesRegex(ValueError,'identical folds'):
                load_bundles(paths)

    def test_saver_rejects_partial_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            saver=PredictionSaver(Path(tmp)/'predictions',Path(tmp)/'log.csv',Path(tmp)/'artifacts')
            with self.assertRaises(ValueError):
                saver.save([1],[.5],.5,{},train_ids=[0])
            with self.assertRaises(ValueError):
                saver.save([1],[np.nan],.5,{})


if __name__=='__main__':
    unittest.main()
