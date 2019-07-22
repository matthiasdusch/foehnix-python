import pytest
import pandas as pd
import numpy as np

from foehnix.foehnix_functions import standardize


@pytest.fixture()
def logitx():

    vals = pd.DataFrame([], columns=['Intercept', 'concomitant'],
                        index=np.arange(50), dtype=float)
    vals.loc[:, 'Intercept'] = 1
    vals.loc[:, 'concomitant'] = np.random.rand(50)
    scale = vals.std()
    center = vals.mean()
    center[scale == 0] = 0
    scale[scale == 0] = 1

    return {'values': vals,
            'scale': scale,
            'center': center,
            'is_standardized': False}


def test_foehnix_functions_standardize(logitx):
    # test wrong inputs
    with pytest.raises(Exception):
        standardize(3)

    # test standardization
    stdlogitx = standardize(logitx)
    assert stdlogitx['is_standardized'] is True
    assert (stdlogitx['scale'] == logitx['scale']).all()
    assert (stdlogitx['center'] == logitx['center']).all()

    np.testing.assert_almost_equal(stdlogitx['scale']['concomitant'],
                                   logitx['values']['concomitant'].std())
    np.testing.assert_almost_equal(stdlogitx['center']['concomitant'],
                                   logitx['values']['concomitant'].mean())

    assert (stdlogitx['values']['concomitant'] !=
            logitx['values']['concomitant']).all()


def test_foehnix_functions_standardize2(logitx):
    # test wrong inputs
    with pytest.raises(Exception):
        standardize(3)

    # test standardization
    stdlogitx = standardize(logitx)
    assert stdlogitx['is_standardized'] is True
    assert (stdlogitx['scale'] == logitx['scale']).all()
    assert (stdlogitx['center'] == logitx['center']).all()

    np.testing.assert_almost_equal(stdlogitx['scale']['concomitant'],
                                   logitx['values']['concomitant'].std())
    np.testing.assert_almost_equal(stdlogitx['center']['concomitant'],
                                   logitx['values']['concomitant'].mean())

    assert (stdlogitx['values']['concomitant'] !=
            logitx['values']['concomitant']).all()
