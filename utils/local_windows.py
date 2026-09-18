"""Cross-fitted local target rates using sorted prefix sums, without pairwise matrices."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class LocalWindowEncoder:
    SPECS = (('Annual_Income_USD', (25, 100, 500), 1),
             ('Daily_Commute_km', (1, 3, 10), 10))

    def __init__(self, cv=5, random_state=42, smooth=20.):
        if smooth <= 0 or not np.isfinite(smooth):
            raise ValueError('smooth must be positive and finite')
        self.cv, self.random_state, self.smooth = cv, random_state, smooth

    @staticmethod
    def values(frame, column, scale):
        values = frame[column].to_numpy(dtype=float)
        return np.round(values * scale) if scale == 10 else values

    def state(self, X, y):
        states = []
        for col, widths, scale in self.SPECS:
            values = self.values(X, col, scale)
            valid = np.isfinite(values)
            order = np.argsort(values[valid], kind='stable')
            states.append((values[valid][order], np.r_[0., np.cumsum(y[valid][order])]))
        return float(y.mean()), states

    def apply(self, X, state):
        prior, states = state
        additions = {}
        for (col, widths, scale), (values, sums) in zip(self.SPECS, states):
            query = self.values(X, col, scale)
            for width in widths:
                left = np.searchsorted(values, query-width*scale, side='left')
                right = np.searchsorted(values, query+width*scale, side='right')
                count = right-left
                total = sums[right]-sums[left]
                missing = ~np.isfinite(query)
                count[missing] = 0; total[missing] = 0
                prefix = f'window_{col}_{width}'
                additions[prefix+'_mean'] = ((total+self.smooth*prior)/(count+self.smooth)).astype('float32')
                additions[prefix+'_log_count'] = np.log1p(count).astype('float32')
        return pd.DataFrame(additions, index=X.index)

    def fit_transform(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isin(y, [0,1]).all():
            raise ValueError('Aligned binary labels required')
        if min((y==0).sum(), (y==1).sum()) < self.cv:
            raise ValueError('Not enough class members for inner folds')
        result = None
        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        for fit, valid in cv.split(X, y):
            block = self.apply(X.iloc[valid], self.state(X.iloc[fit], y[fit]))
            if result is None:
                result = pd.DataFrame(np.nan, index=X.index, columns=block.columns, dtype='float32')
            result.iloc[valid] = block.to_numpy()
        self.state_ = self.state(X, y)
        return result

    def transform(self, X):
        return self.apply(X, self.state_)


def replace_exact_target_features(frame, windows):
    """Replace six exact-income/commute TE columns; retain raw values and other keys."""
    remove = [f'TE_{col}_{smooth}' for col in ('Annual_Income_USD','Daily_Commute_km')
              for smooth in ('auto','10','100')]
    return pd.concat([frame.drop(columns=remove), windows], axis=1)
