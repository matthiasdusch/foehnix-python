import numpy as np
from scipy.stats import logistic
import logging

# logger
log = logging.getLogger(__name__)


def iwls_logit(X, y, beta=None, penalty=None, standardize=True, maxit=100,
               tol=1e-8):
    """Iterative weighted least squares solver for a logistic regression model.

    """
    # TODO check NaNs

    # TODO check constant covariants

    # TODO check standardize

    # Initialize regression coefficients if needed
    if beta is None:
        beta = 0
        beta = np.zeros(2)# .reshape(2,1)
        # TODO ist das so? Nur wenn concomitant max 1

    # TODO besser grundsaetzlich verhindern
    if len(X.shape) == 1:
        X = X.reshape(1, len(X))

    # TODO besser schon frueher
    # y = y.reshape(len(y), 1)

    eta = X.dot(beta)
    eta = eta.reshape(len(eta), 1)
    mu = logistic.cdf(eta)

    # Lists to trace log-likelihood path and the development of
    # the coefficients during EM optimization.
    llpath = []
    coefpath = []

    i = 0  # iteration variable
    delta = 1  # likelihood difference between to iteration: break criteria
    converged = True  # Set to False if we do not converge before maxit
    eps = np.finfo(float).eps

    while delta > tol:

        # new weights
        w = np.sqrt(mu * (1-mu)) + eps
        if penalty is None:
            reg = 0
        else:
            reg = np.diag(np.ones_like(beta)*penalty)
            reg[0, 0] = 0
        beta = np.linalg.inv(((X*w).T).dot(X*w) + reg).dot((X*w).T).dot(
            eta*w + (y-mu) / w)

        # update latent response eta
        eta = X.dot(beta)

        # update response
        mu = logistic.cdf(eta)

        # update log-likelihood sum
        llpath.append(np.sum(y * eta - np.log(1+np.exp(eta))))
        coefpath.append(beta)

        if penalty is None:
            pentext = 'unregularized'
        else:
            pentext = 'lambda = %10.4f' % penalty

        log.info('Iteration %d, ll=%15.4f, %s' % (i, llpath[-1], pentext))

        if i > 0:
            delta = llpath[i] - llpath[i-1]

        # check if we converged
        if i == maxit:
            converged = False
            log.critical('IWLS solver for logistic model did not converge.')
            break

        i += 1

    # If converged, remove last likelihood and coefficient entries
    if converged:
        llpath = llpath[:-1]
        coefpath = coefpath[:-1]

    # TODO: standardized
    beta_vcov = 0
    coef = 0

    beta = coefpath[-1]
    # TODO add column name

    # TODO EDF
    edf = 0

    # unscale coefficients if needed
    ll = llpath[-1]
    rval = {'lambda': penalty,
            'edf': edf,
            'loglik': ll,
            'AIC': -2*ll + 2*edf,
            'BIC': -2*ll + np.log(len(X)) * edf,
            'converged': converged,
            'beta': beta,
            'beta_vcov': beta_vcov,
            'coef': coef}

    return rval
