import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import logging
from copy import deepcopy
import matplotlib.pyplot as plt


def test_plot_api(caplog, tyr_mod1):
    # Test some wrong inputs to the plot API

    with pytest.raises(TypeError) as e:
        tyr_mod1.plot()
    assert e.match('missing 1 required positional argument')

    with pytest.raises(ValueError) as e:
        tyr_mod1.plot(1)
    assert e.match('Argument must be string or list of strings.')

    # will not raise an error but not plot either
    tyr_mod1.plot('foo')
    assert ('Skipping "foo", not a valid plot argument' in
            caplog.records[-1].message)

    tyr_mod1.plot(['foo', 'bar'])
    assert ('Skipping "bar", not a valid plot argument' in
            caplog.records[-1].message)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_loglik(tyr_mod1):
    # with a logarithmic scale
    tyr_mod1.plot('loglik')
    x = plt.gca().get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, np.log(tyr_mod1.optimizer['loglikpath'].index))

    # title
    assert 'foehnix log-likelihood path' == plt.gca().get_title()

    # on the iteration scale
    tyr_mod1.plot('loglik', log=False)
    x = plt.gca().get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, tyr_mod1.optimizer['loglikpath'].index)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_loglikcontri(tyr_mod1):
    # with a logarithmic scale
    tyr_mod1.plot('loglikcontribution')
    x = plt.gca().get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, np.log(tyr_mod1.optimizer['loglikpath'].index))

    # title
    assert 'foehnix log-likelihood contribution' == plt.gca().get_title()

    # on the iteration scale
    tyr_mod1.plot('loglikcontribution', log=False)
    x = plt.gca().get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, tyr_mod1.optimizer['loglikpath'].index)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_coef(tyr_mod1, tyr_mod2):
    # with a logarithmic scale
    tyr_mod1.plot('coef')
    x = plt.gca().get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, np.log(tyr_mod1.optimizer['coefpath'].index))
    # title
    assert 'coefficient path (components)' == plt.gca().get_title()
    # without concomitant -> 1 axes
    assert len(plt.gcf().axes) == 1

    # on the iteration scale with concomitants
    tyr_mod2.plot('coef', log=False)
    x = plt.gcf().axes[0].get_lines()[-1].get_xydata()[:, 0]
    npt.assert_array_equal(x, tyr_mod2.optimizer['coefpath'].index)
    # with concomitant -> 2 axes
    assert len(plt.gcf().axes) == 2


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_histogram(tyr_mod1):
    tyr_mod1.plot('hist')
    # title
    assert 'Conditional Histogram' in plt.gca().get_title()
