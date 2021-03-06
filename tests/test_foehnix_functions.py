import pytest
import pandas as pd
import numpy.testing as npt
from copy import deepcopy
import logging
import numpy as np

from foehnix.foehnix_functions import (standardize, destandardized_values,
                                       destandardized_coefficients)


def test_foehnix_functions_standardize(logitx, caplog):
    # test wrong inputs
    with pytest.raises(Exception):
        standardize(3)
    with pytest.raises(Exception):
        destandardized_values(3)

    # test standardization results
    stdlogitx = deepcopy(logitx)
    standardize(stdlogitx)
    assert stdlogitx['is_standardized'] is True
    assert (stdlogitx['scale'] == logitx['scale']).all()
    assert (stdlogitx['center'] == logitx['center']).all()

    npt.assert_almost_equal(stdlogitx['scale']['concomitantA'],
                                   logitx['values']['concomitantA'].std())
    npt.assert_almost_equal(stdlogitx['center']['concomitantA'],
                                   logitx['values']['concomitantA'].mean())

    assert (stdlogitx['values']['concomitantA'] !=
            logitx['values']['concomitantA']).all()

    # standardize again and catch INFO warning
    caplog.set_level(logging.INFO)
    standardize(stdlogitx)
    assert 'data is already standardized.' in caplog.records[-1].message

    # destandardize again and test results with original data
    dstd_values = destandardized_values(stdlogitx)
    npt.assert_array_almost_equal(dstd_values, logitx['values'])

    # destandardize initial logitx and catch INFO warning
    _ = destandardized_values(logitx)
    assert 'data not standardized' in caplog.records[-1].message


def test_foehnix_functions_destandardize_coefs(random_logitx):
    beta = pd.Series([2., -5, -5], index=random_logitx['values'].columns)
    beta2 = destandardized_coefficients(beta, random_logitx)

    # manually de-scale
    beta3 = beta/random_logitx['scale']
    beta3['Intercept'] = (beta['Intercept'] -
                          np.sum(beta.loc[['concomitantA', 'concomitantB']] /
                                 random_logitx['scale'].loc[['concomitantA',
                                                             'concomitantB']] *
                                 random_logitx['center'].loc[['concomitantA',
                                                              'concomitantB']])
                          )

    npt.assert_array_equal(beta2, beta3)
