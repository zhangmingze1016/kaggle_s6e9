import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.local_windows import LocalWindowEncoder, replace_exact_target_features
from utils.cv_runner import parse_args


class WindowTest(unittest.TestCase):
    def test_prefix_sums_match_direct_inclusive_windows(self):
        X=pd.DataFrame({'Annual_Income_USD':[0.,25.,25.,100.,np.nan],
                        'Daily_Commute_km':[1.,2.,2.,10.,np.nan]})
        y=np.array([0.,1.,0.,1.,0.])
        query=pd.DataFrame({'Annual_Income_USD':[25.,10000.,np.nan],
                            'Daily_Commute_km':[2.,1000.,np.nan]},index=[3,7,9])
        enc=LocalWindowEncoder()
        output=enc.apply(query,enc.state(X,y))
        for col,widths,scale in enc.SPECS:
            for width in widths:
                for i in range(len(query)):
                    center=query[col].iloc[i]
                    mask=(X[col]>=center-width)&(X[col]<=center+width)
                    expected=(y[mask].sum()+20*y.mean())/(mask.sum()+20)
                    self.assertAlmostEqual(output[f'window_{col}_{width}_mean'].iloc[i],expected,places=6)
                    self.assertAlmostEqual(output[f'window_{col}_{width}_log_count'].iloc[i],np.log1p(mask.sum()),places=6)

    def test_crossfit_excludes_self_and_transform_is_stable(self):
        X=pd.DataFrame({'Annual_Income_USD':np.arange(100)*10000.,
                        'Daily_Commute_km':np.arange(100)*100.})
        y=np.tile([0,1],50)
        enc=LocalWindowEncoder()
        values=enc.fit_transform(X,y)
        np.testing.assert_allclose(values.filter(regex='_mean$'),.5)
        np.testing.assert_allclose(values.filter(regex='_log_count$'),0)
        before=enc.transform(X)
        enc.transform(X.iloc[:2]*-100)
        assert_frame_equal(before,enc.transform(X))

    def test_view_replaces_only_exact_target_columns(self):
        frame=pd.DataFrame({'Age':[40],'TE_inc_q100_auto':[.4],**{
            f'TE_{col}_{tag}':[.3] for col in ('Annual_Income_USD','Daily_Commute_km')
            for tag in ('auto','10','100')}})
        out=replace_exact_target_features(frame,pd.DataFrame({'window_test':[.2]}))
        self.assertEqual(out.columns.tolist(),['Age','TE_inc_q100_auto','window_test'])
        self.assertFalse(parse_args('LightGBM',['--preset','strong']).local_windows)
        self.assertTrue(parse_args('LightGBM',['--preset','strong','--local-windows']).local_windows)
