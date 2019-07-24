import pytest
import pandas as pd
import numpy as np

from foehnix.foehnix_functions import (standardize, destandardized_values,
                                       destandardized_coefficients)


@pytest.fixture()
def logitx():
    vals = pd.DataFrame([], columns=['Intercept', 'concomitant'],
                        index=np.arange(50), dtype=float)
    vals.loc[:, 'Intercept'] = 1.5
    vals.loc[:, 'concomitant'] = np.random.rand(50)
    scale = vals.std()
    center = vals.mean()
    center[scale == 0] = 0
    scale[scale == 0] = 1

    return {'values': vals,
            'scale': scale,
            'center': center,
            'is_standardized': False}


@pytest.fixture()
def logitx_2():
    vals = pd.DataFrame([], columns=['Intercept', 'A', 'B'],
                        index=np.arange(500), dtype=float)
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

    standardize(stdmatrix)

    stdmatrix['center']['Intercept'] = 10
    stdmatrix['center']['A'] = 5
    stdmatrix['center']['B'] = -20
    stdmatrix['scale']['Intercept'] = 1
    stdmatrix['scale']['A'] = 2
    stdmatrix['scale']['B'] = 5

    return stdmatrix


def test_foehnix_functions_standardize(logitx):
    # test wrong inputs
    with pytest.raises(Exception):
        standardize(3)
        destandardized_values(3)

    # test standardization results
    stdlogitx = logitx.copy()
    standardize(stdlogitx)
    assert stdlogitx['is_standardized'] is True
    assert (stdlogitx['scale'] == logitx['scale']).all()
    assert (stdlogitx['center'] == logitx['center']).all()

    np.testing.assert_almost_equal(stdlogitx['scale']['concomitant'],
                                   logitx['values']['concomitant'].std())
    np.testing.assert_almost_equal(stdlogitx['center']['concomitant'],
                                   logitx['values']['concomitant'].mean())

    assert (stdlogitx['values']['concomitant'] !=
            logitx['values']['concomitant']).all()

    # destandardize again and test results with original data
    dstd_values = destandardized_values(stdlogitx)
    np.testing.assert_array_almost_equal(dstd_values, logitx['values'])


def test_foehnix_functions_destandardize_coefs(logitx_2):
    beta = pd.Series([2., -5, -5], index=logitx_2['values'].columns)
    beta2 = destandardized_coefficients(beta, logitx_2)
    # TODO: this needs some attention first
    # np.testing.assert_array_almost_equal(beta2, beta/logitx_2['scale'])
