import numpy as np
from scipy.stats import logistic
import logging
import pandas as pd

import foehnix.foehnix_functions as func

# logger
log = logging.getLogger(__name__)

# TODO: Think about making iwls_logit a class ccmodel with an method iwls_logit


def iwls_logit(logitx, y, beta=None, penalty=None, standardize=True, maxit=100,
               tol=1e-8):
    """
    Iterative weighted least squares solver for a logistic regression model.

    Parameters
    ----------
    logitx : dict
        Must contain:

        - ``'values'`` : :py:class:`numpy.ndarray` the model matrix
        - ``'center'`` : list, containing the mean of each model matrix row
        - ``'scale'`` : list, containing the standard deviation of matrix rows
        - ``'name'`` : list, containing the names of the rows
        - ``'is_standardized'``: boolean info if matrix is standardized
    y : :py:class:`numpy.ndarray`
        predictor values of shape(len(observations), 1)
    beta : :py:class:`numpy.ndarray`
        initial regression coefficients. If None will be initialized with 0.
    penalty : float
        Penalty for regularization. Default is None.
    standardize : bool
        If True (default) the model matrix will be standardized
    maxit : int
        maximum number of iterations, default 100.
    tol : float
        tolerance for improvement between iterations, default 1e-8.

    Returns
    -------
    : dict


    """
    # do we have to standardize the model matrix?
    # TODO: do we actually call iwls_logit with standardize=True somewhere?
    if standardize is True:
        logitx = func.standardize(logitx)

    x = logitx['values']

    if np.isnan(x).any():
        raise ValueError('Input logitx.values contains NaN!')
    if np.isnan(y).any():
        raise ValueError('Input y contains NaN!')

    # check if we have columns with constant values (except one intercept).
    if [len(np.unique(x[:, col])) for col in range(x.shape[1])].count(1) > 1:
        raise ValueError('Model matrix contains columns with constant values.')

    # y must be within 0 and 1
    if (min(y) < 0) or (max(y) > 1):
        raise ValueError('Values of y must be within ]0, 1[.')

    # Initialize regression coefficients if needed
    if beta is None:
        beta = np.zeros(x.shape[1])

    eta = x.dot(beta)
    eta = eta.reshape(len(eta), 1)
    prob = logistic.cdf(eta)

    # Lists to trace log-likelihood path and the development of
    # the coefficients during optimization.
    llpath = []
    coefpath = []

    i = 1  # iteration variable
    delta = 1  # likelihood difference between to iteration: break criteria
    converged = True  # Set to False if we do not converge before maxit
    eps = np.finfo(float).eps

    while delta > tol:

        # new weights
        w = np.sqrt(prob * (1-prob)) + eps

        # TODO reg is constant. Or will it change in later versions?
        if penalty is None:
            reg = 0
        else:
            reg = np.diag(np.ones_like(beta)*penalty)
            reg[0, 0] = 0

        beta = np.linalg.inv((x*w).T.dot(x*w) + reg).dot((x*w).T).dot(
            eta*w + (y-prob) / w)

        # update latent response eta
        eta = x.dot(beta)

        # update response
        prob = logistic.cdf(eta)

        # update log-likelihood sum
        llpath.append(np.sum(y * eta - np.log(1+np.exp(eta))))
        coefpath.append(beta)

        if penalty is None:
            pentext = 'unregularized'
        else:
            pentext = 'lambda = %10.4f' % penalty

        log.debug('Iteration %d, ll=%15.4f, %s' % (i, llpath[-1], pentext))

        if i > 1:
            delta = llpath[-1] - llpath[-2]

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

    # calculate standard error
    if logitx['is_standardized'] is True:
        xds = func.destandardize(logitx.copy())['values']
    else:
        xds = x
    beta_se = np.sqrt(np.diag(np.linalg.inv((xds*w).T.dot(xds*w))))
    del xds

    beta = coefpath[-1]

    # TODO reg is still the same as within the while loop!?
    if penalty is None:
        reg = 0
    else:
        reg = np.diag(np.ones_like(beta)*penalty)
        reg[0, 0] = 0

    # Effective degree of freedom
    edf = np.sum(np.diag((x*w).T.dot(x*w).dot(
        np.linalg.inv((x*w).T.dot(x*w) + reg))))

    # TODO Reto:
    # rval$coef <- if ( standardize ) destandardize_coefficients(beta, X)
    #              else beta
    # aber später nocham destandardize??!??
    # und again: standardize hier immer False??
    # Meine Idee: coef hier immer destandardized
    # TODO mit Reto besprechen
    """
    if standardize is True:
        coef = func.destandardize_coefficients(beta, logitx)
    else:
        coef = beta
    """

    # Keep coefficients destandardized
    if logitx['is_standardized'] is True:
        coef = func.destandardize_coefficients(beta.copy(), logitx)
    else:
        coef = beta.copy()

    coef = dict(zip(list(logitx['names']), coef.squeeze()))
    # TODO und beta==coef

    # TODO evtl beta oder coef als dict uebergeben:
    # TODO  beta/coef = dict(zip(list(logitx['names']), beta.squeeze()))

    # final logliklihood
    ll = llpath[-1]

    rval = {'lambda': penalty,
            'edf': edf,
            'loglik': ll,
            'AIC': -2*ll + 2*edf,
            'BIC': -2*ll + np.log(len(x)) * edf,
            'converged': converged,
            'beta': beta,
            'beta_se': beta_se,
            'coef': coef,
            'iter': i-1}

    return rval


def iwls_summary(ccmodel):
    """
    Prints some statistics for a given concomitant model

    Parameters
    ----------
    ccmodel : dict
        which is returnded by :py:class:`foehnix.iwls_logit`
    """

    print('\nConcomitant model: z test of coefficients')

    # TODO bei conomitant model z test sind Retos coefs standardized. Soll so?

    print('Number of IWLS iterations %d (%s)' %
          (ccmodel['iter'],
           ('converged' if ccmodel['converged'] else 'not converged')))
    print("Dispersion parameter for binomial family taken to be 1.")