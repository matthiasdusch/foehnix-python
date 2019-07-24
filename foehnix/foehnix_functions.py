import numpy as np
import logging

# logger
log = logging.getLogger(__name__)


def standardize(x):
    """
    Function to standardize the values of the concomitant matrix.

    Parameters
    ----------
    x : dict
        Must contain:
        - ``'values'`` : :py:class:`pandas.DataFrame` the model matrix
        - ``'center'`` : :py:class:`pandas.Series`, containing the mean of each
          model matrix row
        - ``'scale'`` : :py:class:`pandas:Series`, containing the standard
          deviation of matrix rows
        - ``'is_standardized'``: bool, will trigger standardization if False
          and will be set to True afterwards
    """
    if x['is_standardized'] is False:
        x['values'] = (x['values'] - x['center']) / x['scale']
        x['is_standardized'] = True
        log.debug('Model matrix standardized.')
    else:
        log.info('Standardization called but data is already standardized.')


def destandardized_values(stdx):
    """
    Function returns the DE-standardize values of the concomitant matrix.

    Parameters
    ----------
    stdx : dict
        Must contain:
        - ``'values'`` : :py:class:`pandas.DataFrame` the model matrix
        - ``'center'`` : :py:class:`pandas.Series`, containing the mean of each
          model matrix row
        - ``'scale'`` : :py:class:`pandas:Series`, containing the standard
          deviation of matrix rows
        - ``'is_standardized'``: bool, will trigger destandardization if True
          and will be set to False afterwards

    Returns
    -------
    destdx : :py:class:`numpy.ndarray`
        the
    """
    if stdx['is_standardized'] is True:
        destdx = (stdx['values'] * stdx['scale'] + stdx['center']).values
        log.debug('Model matrix destandardized.')
    else:
        destdx = stdx['values'].values
        log.info('Trying to destandardize values but data not standardized.')

    return destdx


def destandardized_coefficients(beta, x):
    """
    Returns DE-standardizes the Regression Coefficients

    Brings coefficients back to the "real" scale if standardized coefficients
    are used when estimating the logistic regression model (concomitant model).

    Parameters
    ----------
    beta : :py:class:`pandas.Series`
        regression coefficients
    x : dict
        Must contain:
        - ``'values'`` : :py:class:`numpy.ndarray` the model matrix
        - ``'center'`` : list, containing the mean of each model matrix row
        - ``'scale'`` : list, containing the standard deviation of matrix rows

    Returns
    -------
    destdbeta : :py:class:`pandas.Series`
        destandardized regression coefficients
    """
    destdbeta = beta.copy()

    if 'Intercept' in beta:
        nic = destdbeta.index[destdbeta.index != 'Intercept']
        # Descaling intercept
        destdbeta['Intercept'] = destdbeta['Intercept'] - np.sum(
            destdbeta[nic] * x['center'][nic] / x['scale'][nic])
        # Descaling all other regression coefficients
        destdbeta[nic] = destdbeta[nic] / x['scale'][nic]

        log.debug('Regression coefficients destandardized (with Intercept).')
    else:
        destdbeta = destdbeta / x['scale']
        log.debug('Regression coefficients destandardized (no Intercept).')

    return destdbeta
