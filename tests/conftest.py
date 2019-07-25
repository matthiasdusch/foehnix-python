import pytest
import pandas as pd
import numpy as np

from foehnix.foehnix_functions import standardize


@pytest.fixture(scope="function")
def logitx(data):
    """
    Pytest fixture for a concomitant matrix dictionary

    Time is not an issue here, so use scope=function instead of e.g session

    Returns
    -------
    dict

    """
    ix = np.arange(len(data))
    vals = pd.DataFrame([],
                        columns=['Intercept', 'concomitantA', 'concomitantB'],
                        index=ix, dtype=float)

    vals.loc[ix, 'Intercept'] = 1
    vals.loc[ix, 'concomitantA'] = data.loc[:, 'rh'].values
    vals.loc[ix, 'concomitantB'] = data.loc[:, 'rand'].values

    scale = vals.std()
    center = vals.mean()
    # If std == 0 (e.g. for the Intercept), set center=0 and scale=1
    center[scale == 0] = 0
    scale[scale == 0] = 1

    x = {'values': vals,
         'scale': scale,
         'center': center,
         'is_standardized': False}
    return x


@pytest.fixture(scope="function")
def random_logitx(logitx):
    """
    vals.loc[:, 'Intercept'] = np.random.normal(10, 2, 500)
    vals.loc[:, 'A'] = np.random.normal(5, 3, 500)
    vals.loc[:, 'B'] = np.random.normal(-5, 1, 500)
    scale = vals.std()
    center = vals.mean()
    center[scale == 0] = 0
    scale[scale == 0] = 1

    stdmatrix = {'values': vals,
                 'scale': scale,
                 'center': center,
                 'is_standardized': False}
    """

    standardize(logitx)

    logitx['center']['Intercept'] = 10
    logitx['center']['concomitantA'] = 5
    logitx['center']['concomitantB'] = -20
    logitx['scale']['Intercept'] = 1
    logitx['scale']['concomitantA'] = 2
    logitx['scale']['concomitantB'] = 5

    return logitx


@pytest.fixture(scope="session")
def data():
    """
    n = 100
    data = pd.DataFrame([], columns=['predictor', 'concomitantA',
                                     'concomitantB'],
                        index=np.arange(n), dtype=float)
    data.loc[:, 'predictor'] = np.random.normal(10, 2, n)
    data.loc[:, 'concomitantA'] = np.random.normal(5, 3, n)
    data.loc[:, 'concomitantB'] = np.random.normal(-5, 1, n)
    """

    # Set up my data
    np.random.seed(1)

    # data points
    n = 100
    # make two wind speed clusters
    mu1 = 10
    mu2 = 25
    sd1 = 4
    sd2 = 7

    # reasonable windspeed cluster as predictor
    ff = np.random.normal(loc=mu1, scale=sd1, size=int(3*n/4))
    ff = np.append(ff, np.random.normal(loc=mu2, scale=sd2, size=int(n/4)))

    # relative humidity as concomitantA
    rh = 70 - ff + np.random.normal(loc=0, scale=5, size=n)

    # some random stuff as concomitantB
    cc_b = np.random.normal(loc=-5, scale=1, size=n)

    # wind direction with some NaNs
    dd = np.random.randint(0, 360, n).astype(float)
    dd[np.random.randint(0, n, int(n/10))] = np.nan

    # make a pandas data frame
    data = pd.DataFrame({'ff': ff,
                         'rh': rh,
                         'dd': dd,
                         'rand': cc_b})

    return data


@pytest.fixture(scope="function")
def model_response(data):
    y = data.loc[:, 'ff'].values.copy()
    y = y.reshape(len(y), 1)

    z = np.zeros_like(y)
    z[y >= np.mean(y)] = 1

    return z
