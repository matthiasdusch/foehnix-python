import numpy as np
import scipy
import logging

# logger
log = logging.getLogger(__name__)


class Family:
    """
    Common logic for the foehnix families

    """

    def __init__(self):
        self.name = 'Main family'

    def density(self, y, mu, sigma):
        raise NotImplementedError

    def distribution(self, q, mu):
        raise NotImplementedError

    def loglik(self, y, post, prob, theta):
        raise NotImplementedError

    def random_sample(self, n, mu, sigma):
        raise NotImplementedError

    def posterior(self, y, prob, theta):
        raise NotImplementedError

    def theta(self, y, post, init=False):
        raise NotImplementedError


class GaussianFamily(Family):
    """
    Gaussian foehnix mixture model family

    """

    def __init__(self):
        """
        Initialize the Gaussian Family
        """
        self.name = 'Gaussian'

    def density(self, y, mu, sigma):
        raise NotImplementedError

    def distribution(self, q, mu):
        raise NotImplementedError

    def loglik(self, y, post, prob, theta):
        """
        Calculate log-likelihood sum of the two-component mixture model

        Parameters
        ----------
        post : py:class:`numpy.array`
            posteriori
        prob : py:class:`numpy.array`
            probability
        theta : dict
            contains mu1, mu2, logsd1, logsd2

        Returns
        -------
        : dict
            Component, concomitant and sum of both

        """
        # limit prob to [eps, 1-eps]
        eps = np.sqrt(np.finfo(float).eps)
        prob = np.maximum(eps, np.minimum(1-eps, prob))

        # calculate log liklihood

        dnorm1 = scipy.stats.norm(loc=theta['mu1'],
                                  scale=np.exp(theta['logsd1'])).logpdf(y)
        dnorm2 = scipy.stats.norm(loc=theta['mu2'],
                                  scale=np.exp(theta['logsd2'])).logpdf(y)

        component = np.sum(post * dnorm2) + np.sum((1-post) * dnorm1)

        concomitant = np.sum((1-post) * np.log(1-prob) + post * np.log(prob))

        return {'component': component,
                'concomitant': concomitant,
                'full': component+concomitant}

    def random_sample(self, n, mu, sigma):
        raise NotImplementedError

    def posterior(self, y, prob, theta):
        dnorm1 = scipy.stats.norm(loc=theta['mu1'],
                                  scale=np.exp(theta['logsd1'])).pdf(y)
        dnorm2 = scipy.stats.norm(loc=theta['mu2'],
                                  scale=np.exp(theta['logsd2'])).pdf(y)

        post = prob * dnorm2 / ((1-prob) * dnorm1 + prob * dnorm2)
        return post

    def theta(self, y, post, init=False):
        # Emperical update of mu and std
        mu1 = np.sum((1-post) * y) / (np.sum(1-post))
        mu2 = np.sum(post * y) / np.sum(post)

        if init:
            sd1 = np.std(y)
            sd2 = np.std(y)
        else:
            sd1 = np.sqrt(np.sum((1-post) * (y - mu1)**2) / np.sum(1-post))
            sd2 = np.sqrt(np.sum((post) * (y - mu2)**2) / np.sum(post))

        # return dict
        theta = {'mu1': mu1,
                 'logsd1': np.log(sd1) if sd1 > np.exp(-6) else -6,
                 'mu2': mu2,
                 'logsd2': np.log(sd2) if sd2 > np.exp(-6) else -6}
        return theta


def initialize_family(familyname='gaussian', left=float('-Inf'),
                      right=float('Inf'), truncated=False):
    """
    Helper function to initialize a Foehnix Family based on arguments

    Parameters
    ----------
    familyname : str

        - `gaussian' (default)
        - 'logistic'
    truncated : bool
    left : float
    right : float

    Returns
    -------
    py:class:`foehnix.Family` object

    """

    if not isinstance(truncated, bool):
        raise ValueError('truncated must be boolean True or False')

    if familyname == 'gaussian':
        if np.isfinite([left, right]).any():
            if truncated is True:
                family = TruncatedGaussianFamily(left=left, right=right)
            else:
                family = CensoredGaussianFamily(left=left, right=right)
        else:
            family = GaussianFamily()

    elif familyname == 'logistic':
        if np.isfinite([left, right]).any():
            if truncated is True:
                family = TruncatedLogisticFamily(left=left, right=right)
            else:
                family = CensoredLogsticFamily(left=left, right=right)
        else:
            family = LogisticFamily()
    else:
        raise ValueError('familyname must be gaussian or logistic')

    log.debug('%s model family initialized.' % family.name)
    return family
