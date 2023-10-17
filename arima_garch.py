import numpy as np
from scipy import optimize
from tqdm.notebook import tqdm
import math
from scipy.stats import t

from arch.univariate import ARX
from arch.univariate import StudentsT

from arch.univariate import GARCH
import pandas as pd
def rolling_arima_garch(data, window_size, refit_every, prediction_horizon, p, q):

    forecast_mean = pd.DataFrame([])
    forecast_variance = pd.DataFrame([])
    Value_at_risk = pd.DataFrame([])
    nu = []
    for i in tqdm(range(math.ceil((len(data.returns.values) - window_size +1)/refit_every))):


        ar = ARX(data.returns[i*refit_every:(i+1)*refit_every + window_size - 1], lags=[1], rescale=True)

        ar.volatility = GARCH(p=p, q = q)
        ar.distribution = StudentsT()
        res = ar.fit(update_freq=0, disp="off", last_obs = window_size)

        forecast = res.forecast(horizon = prediction_horizon, reindex=True, align = 'origin')
        Value_at_risk = pd.concat([Value_at_risk, (forecast.mean / res.scale).loc[i*refit_every+window_size - 1:]+((forecast.variance / res.scale**2)**0.5 ).loc[i*refit_every+window_size - 1:]* t.ppf(0.05, res.params.nu)])

        forecast_mean = pd.concat([ forecast_mean, (forecast.mean / res.scale).loc[i*refit_every+window_size - 1:]])

        forecast_variance = pd.concat([forecast_variance, ((forecast.variance / res.scale**2)**0.5 ).loc[i*refit_every+window_size - 1:]])
        nu.append(res.params.nu)

        
    return forecast_mean, forecast_variance, Value_at_risk, nu