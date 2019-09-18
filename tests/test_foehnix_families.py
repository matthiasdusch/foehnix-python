import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from scipy.stats import logistic, norm

from foehnix import families


# test common foehnix Family logic
def test_common_logic():

    # init a Family superclass
    fam1 = families.Family()
    assert fam1.name == 'Main family'
    assert fam1.scale_factor is None
    # density is not implemented in Main family
    with pytest.raises(NotImplementedError):
        fam1.density(0, 1, 2)


def test_truncated_censored():
    # both truncated and censord families are not implemented at the moment
    # gaussian censored
    with pytest.raises(NotImplementedError):
        families.initialize_family('gaussian', left=1)
    # gaussian truncated
    with pytest.raises(NotImplementedError):
        families.initialize_family('gaussian', right=1, truncated=True)

    # logistic censored
    with pytest.raises(NotImplementedError):
        families.initialize_family('logistic', right=-1)
    # logistic truncated
    with pytest.raises(NotImplementedError):
        families.initialize_family('logistic', left=-1, truncated=True)


def test_gaussian_distribution(predictor):
    y = deepcopy(predictor)

    # initialize a Gaussian Mixture Model distribution
    gaus = families.initialize_family('gaussian')

    # random mu and sigma
    mu = 1
    sigma = 2

    # calculate density
    dnorm = gaus.density(y, mu, sigma)
    dnorm1 = norm(loc=mu, scale=sigma).pdf(y)
    npt.assert_array_equal(dnorm, dnorm1)

    # also test log-density
    ldnorm = gaus.density(y, mu, sigma, logpdf=True)
    npt.assert_array_equal(ldnorm, norm(loc=mu, scale=sigma).logpdf(y))

    # make a probability
    prob = np.exp(y) / (1 + np.exp(y))

    # calculate posterior
    dnorm2 = norm(loc=5, scale=4).pdf(y)
    post = gaus.posterior(y, prob, {'mu1': 1, 'logsd1': np.log(2),
                                    'mu2': 5, 'logsd2': np.log(4)})

    post1 = prob * dnorm2 / ((1-prob) * dnorm1 + prob * dnorm2)
    npt.assert_array_equal(post, post1)

    # calculate theta
    theta = gaus.theta(y, post)
    theta1 = gaus.theta(y, post, init=True)

    npt.assert_equal(theta['mu1'], theta1['mu1'])
    npt.assert_equal(theta['mu2'], theta1['mu2'])
    npt.assert_equal(theta1['logsd1'], theta1['logsd2'])
    assert theta['logsd1'] != theta['logsd2']

    # calculate log-liklihod
    loli = gaus.loglik(y, post, prob, theta)

    npt.assert_equal(loli['component'] + loli['concomitant'],
                     loli['full'])


def test_logistic_distribution(predictor):
    y = deepcopy(predictor)

    # initialize Logistic Mixture Model distribution
    logi = families.initialize_family('logistic')

    # random mu, sigma and probability
    mu = 1
    sigma = 2
    prob = np.exp(y) / (1 + np.exp(y))

    # calculate density
    dlogi = logi.density(y, mu, sigma)
    dlogi1 = logistic(loc=mu, scale=sigma).pdf(y)
    npt.assert_array_equal(dlogi, dlogi1)

    # also test log-density
    ldlogi = logi.density(y, mu, sigma, logpdf=True)
    npt.assert_array_equal(ldlogi, logistic(loc=mu, scale=sigma).logpdf(y))

    # calculate posterior
    dlogi2 = logistic(loc=5, scale=4).pdf(y)
    post = logi.posterior(y, prob, {'mu1': 1, 'logsd1': np.log(2),
                                    'mu2': 5, 'logsd2': np.log(4)})

    post1 = prob * dlogi2 / ((1-prob) * dlogi1 + prob * dlogi2)
    npt.assert_array_equal(post, post1)

    # calculate theta
    theta = logi.theta(y, post)
    theta1 = logi.theta(y, post, init=True)

    npt.assert_equal(theta['mu1'], theta1['mu1'])
    npt.assert_equal(theta['mu2'], theta1['mu2'])
    npt.assert_equal(theta1['logsd1'], theta1['logsd2'])
    assert theta['logsd1'] != theta['logsd2']

    # calculate log-liklihod
    loli = logi.loglik(y, post, prob, theta)

    npt.assert_equal(loli['component'] + loli['concomitant'],
                     loli['full'])
