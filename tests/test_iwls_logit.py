import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy

from foehnix import iwls_logit
from foehnix.iwls_logit import iwls_summary


def test_wrong_input(logitx, model_response):
    # test NaNs in concomitant
    logitx['values']['concomitantA'][10] = np.nan
    with pytest.raises(Exception) as e:
        iwls_logit(logitx, model_response)
    assert e.match('logitx.values contains NaN')
    logitx['values']['concomitantA'][10] = 0

    # test NaNs in model response
    model_response[10] = np.nan
    with pytest.raises(Exception) as e:
        iwls_logit(logitx, model_response)
    assert e.match('y contains NaN')

    # test model response > 1
    model_response[10] = 2
    with pytest.raises(Exception) as e:
        iwls_logit(logitx, model_response)
    assert e.match('y must be within')
    model_response[10] = 0

    # test constant concomitant
    logitx['values']['concomitantA'] = 23
    with pytest.raises(Exception) as e:
        iwls_logit(logitx, model_response)
    assert e.match('columns with constant values')


def test_iwls_logit_model(logitx, model_response, caplog):
    # model call without standardization
    ccm1 = iwls_logit(deepcopy(logitx), model_response, standardize=False)
    # model call with standardization
    ccm2 = iwls_logit(deepcopy(logitx), model_response, standardize=True)
    # model call which will not converge and throw a warning
    ccm3 = iwls_logit(deepcopy(logitx), model_response, tol=-np.Inf)

    # test convergence
    assert ccm1['converged']
    assert ccm2['converged']
    assert ccm3['converged'] is False
    # last iwls_logit call should also raise a critical logging warning
    assert 'did not converge' in caplog.records[-1].message

    # test iterations
    assert (ccm1['iter'] == ccm1['iter']) and (ccm1['iter'] <= 10)
    assert (ccm3['iter'] == 99)

    # most results should be comparable
    for var in ['AIC', 'BIC', 'beta_se', 'coef', 'edf', 'loglik']:
        npt.assert_array_almost_equal(ccm1[var], ccm2[var])
        # be a little bit more relaxed on the none converged model
        npt.assert_array_almost_equal(ccm1[var], ccm3[var], decimal=3)

    # standardization will result in different beta values
    with pytest.raises(AssertionError):
        npt.assert_almost_equal(ccm1['beta'], ccm2['beta'], decimal=1)

    # ccm3 should be standardized
    npt.assert_almost_equal(ccm2['beta'], ccm3['beta'], decimal=3)

    # but if not standardized, beta and coef should be equal
    npt.assert_array_almost_equal(ccm1['beta'].squeeze(), ccm1['coef'])


def test_iwls_summary(logitx, model_response, capfd):
    ccmodel = iwls_logit(logitx, model_response)
    iwls_summary(ccmodel)

    out, err = capfd.readouterr()
    assert 'cc.Intercept     %0.0f' % ccmodel['coef']['Intercept'] in out
    assert 'cc.concomitantA  %0.0f' % ccmodel['coef']['concomitantA'] in out
    assert 'cc.concomitantB  %0.0f' % ccmodel['coef']['concomitantB'] in out
    assert 'Number of IWLS iterations %d (converged)' % ccmodel['iter'] in out
