import numpy as np
import pandas as pd
from scipy import stats

class Backtest:
    def __init__(self, actual, forecast, alpha):
        self.index = actual.index
        self.actual = actual.values
        self.forecast = forecast.values
        self.alpha = alpha
        self.hit_series = (self.actual < self.forecast) * 1

    def losses(self, delta=25, c=1):
        
        df = pd.DataFrame([self.hit_series.mean(), 
                      ((self.alpha - self.hit_series) * (self.actual - self.forecast)).mean(),
                      (((self.alpha - (1 + np.exp(delta*(self.actual - self.forecast)))**-1) * (self.actual - self.forecast))).mean(),
                      ((self.hit_series * (1 + (self.actual - self.forecast)**2))).mean(),
                      ((self.hit_series * (1 + (self.actual - self.forecast)**2) - c*(1-self.hit_series) * self.forecast)).mean()])
        df.index = ["Hit rate", "Tick loss", "Smooth loss", "Quadratic loss", "Firm loss"]
        df.columns = ['Losses']
        return df

    def Christoffersen_test(self):
        hits = self.hit_series
        tr = hits[1:] - hits[:-1] 

        n01, n10 = (tr == 1).sum(), (tr == -1).sum()
        n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()
        n0, n1 = n01 + n00, n10 + n11
        n = n0 + n1
        p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
        p = n1 / n

        uc_h0 = n0 * np.log(1 - self.alpha) + n1 * np.log(self.alpha)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        ind_h0 = (n00 + n01) * np.log(1 - p+1e-10) + (n01 + n11) * np.log(p+1e-10)
        ind_h1 = n00 * np.log(1 - p01+1e-10) + n01 * np.log(p01 +1e-10) + n10 * np.log(1 - p11+1e-10) +n11 * np.log(p11+1e-10)
        ind = -2 * (ind_h0 - ind_h1)

        cc = uc + ind

        df = pd.concat([pd.Series([uc, ind, cc]),
                        pd.Series([1 - stats.chi2.cdf(uc, 1),
                                    1 - stats.chi2.cdf(ind, 1),
                                    1 - stats.chi2.cdf(cc, 2)])], axis=1)

        # Assign names
        df.columns = ["Statistic", "p-value"]
        df.index = ["Unconditional", "Independence", "Conditional"]

        return df