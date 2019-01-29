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
        self.scale_factor = None

    def density(self, y, mu, sigma, log=False):
        raise NotImplementedError

    # def distribution(self, q, mu):
    #    raise NotImplementedError

    def loglik(self, y, post, prob, theta):
        """
        Calculate log-likelihood sum of the two-component mixture model

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            predictor values of shape(len(observations), 1)
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

        # calculate densities, logistic/gaussian specific
        d1 = self.density(y, theta['mu1'], np.exp(theta['logsd1']), log=True)
        d2 = self.density(y, theta['mu2'], np.exp(theta['logsd2']), log=True)

        # calculate log liklihood
        component = np.sum(post * d2) + np.sum((1-post) * d1)

        concomitant = np.sum((1-post) * np.log(1-prob) + post * np.log(prob))

        return {'component': component,
                'concomitant': concomitant,
                'full': component+concomitant}

    # def random_sample(self, n, mu, sigma):
    #     raise NotImplementedError

    def posterior(self, y, prob, theta):
        """
        Posterior probabilities used for model estimation (EM algorithm)

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            predictor values of shape(len(observations), 1)
        prob : py:class:`numpy.array`
            probability
        theta : dict
            contains mu1, mu2, logsd1, logsd2

        Returns
        -------
        :py:class:`numpy.ndarray`
            (updated) posterior probabilites
        """
        # calculate densities, logistic/gaussian specific
        d1 = self.density(y, theta['mu1'], np.exp(theta['logsd1']), log=False)
        d2 = self.density(y, theta['mu2'], np.exp(theta['logsd2']), log=False)

        post = prob * d2 / ((1-prob) * d1 + prob * d2)
        return post

    def theta(self, y, post, init=False):
        """
        Distribution parameters of the components of the mixture models.

        Used for model estimation in the EM algorithm.

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            predictor values of shape(len(observations), 1)
        post : py:class:`numpy.array`
            posteriori
        init : bool
            If True (first call) scale is just the standard deviation of y.

        Returns
        -------
        : dict
            contains mu1, mu2, logsd1, logsd2
        """
        # Emperical update of mu and std
        mu1 = np.sum((1-post) * y) / (np.sum(1-post))
        mu2 = np.sum(post * y) / np.sum(post)

        if init:
            sd1 = np.std(y)
            sd2 = np.std(y)
        else:
            sd1 = np.sqrt(np.sum((1-post) * (y - mu1)**2) / np.sum(1-post))
            sd2 = np.sqrt(np.sum(post * (y - mu2)**2) / np.sum(post))

        # necessary for the logistic distribution
        sd1 *= self.scale_factor
        sd2 *= self.scale_factor

        # return dict
        theta = {'mu1': mu1,
                 'logsd1': np.log(sd1) if sd1 > np.exp(-6) else -6,
                 'mu2': mu2,
                 'logsd2': np.log(sd2) if sd2 > np.exp(-6) else -6}
        return theta


class GaussianFamily(Family):
    """
    Gaussian foehnix mixture model family

    """

    def __init__(self):
        """
        Initialize the Gaussian Family
        """
        super(Family, self).__init__()

        self.name = 'Gaussian'
        self.scale_factor = 1  # factor for the scale of the distribution

    def density(self, y, mu, sigma, log=False):
        """
        Density function of the mixture distribution

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            predictor values of shape(len(observations), 1)
        mu : float
            location of the distribution
        sigma : float
            scale of the distribution
        log : bool
            If True, log of the probability density function will be returned.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Probability density function or log of it.
        """
        if log is True:
            dnorm = scipy.stats.norm(loc=mu, scale=sigma).logpdf(y)
        else:
            dnorm = scipy.stats.norm(loc=mu, scale=sigma).pdf(y)

        return dnorm


class LogisticFamily(Family):
    """
    Logistic foehnix mixture model family

    """

    def __init__(self):
        """
        Initialize the Logistic Family
        """
        super(Family, self).__init__()

        self.name = 'Logistic'
        self.scale_factor = np.sqrt(3)/np.pi  # distribution scale factor

    def density(self, y, mu, sigma, log=False):
        """
        Density function of the logistic mixture model distribution

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            predictor values of shape(len(observations), 1)
        mu : float
            location of the distribution
        sigma : float
            scale of the distribution
        log : bool
            If True, log of the probability density function will be returned.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Probability density function or log of it.
        """
        if log is True:
            dlogis = scipy.stats.logistic(loc=mu, scale=sigma).logpdf(y)
        else:
            dlogis = scipy.stats.logistic(loc=mu, scale=sigma).pdf(y)

        return dlogis


def initialize_family(familyname='gaussian', left=float('-Inf'),
                      right=float('Inf'), truncated=False):
    """
    Helper function to initialize a Foehnix Family based on arguments

    Parameters
    ----------
    familyname : str
        Gaussian or Logistic distribution. Possible values:

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
