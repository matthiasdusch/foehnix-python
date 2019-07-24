import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
import logging

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

    np.testing.assert_almost_equal(stdlogitx['scale']['concomitantA'],
                                   logitx['values']['concomitantA'].std())
    np.testing.assert_almost_equal(stdlogitx['center']['concomitantA'],
                                   logitx['values']['concomitantA'].mean())

    assert (stdlogitx['values']['concomitantA'] !=
            logitx['values']['concomitantA']).all()

    # standardize again and catch INFO warning
    caplog.set_level(logging.INFO)
    standardize(stdlogitx)
    assert 'data is already standardized.' in caplog.records[-1].message

    # destandardize again and test results with original data
    dstd_values = destandardized_values(stdlogitx)
    np.testing.assert_array_almost_equal(dstd_values, logitx['values'])


def test_foehnix_functions_destandardize_coefs(random_logitx):
    beta = pd.Series([2., -5, -5], index=random_logitx['values'].columns)
    beta2 = destandardized_coefficients(beta, random_logitx)
    # TODO: this needs some attention first
    # np.testing.assert_array_almost_equal(beta2, beta/logitx_2['scale'])
