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

    Returns
    -------

    x : dict
        Same dict, with standardized values if necessary
    """
    if x['is_standardized'] is False:
        std_x = x.copy()
        std_x['values'] = (x['values'] - x['center']) / x['scale']
        std_x['is_standardized'] = True
        log.debug('Model matrix standardized.')
    else:
        log.info('Standardization called but data is already standardized.')

    return std_x


def destandardize(x):
    """
    Function to DE-standardize the values of the concomitant matrix.

    Parameters
    ----------
    x : dict
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

    x : dict
        Same dict, with destandardized values if necessary
    """
    if x['is_standardized'] is True:
        x['values'] = x['values'] * x['scale'] + x['center']
        x['is_standardized'] = False
        log.debug('Model matrix destandardized.')
    else:
        log.info('Destandardization called but data is not standardized.')

    return x


def destandardize_coefficients(beta, x):
    """
    Destandardize Regression Coefficients

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
        - ``'name'`` : list, containing the names of the rows
        - ``'is_standardized'``: bool, will trigger standardization if False
          and will be set to True afterwards

    Returns
    -------

    x : list
        Destandardized regression coefficients
    """
    if 'Intercept' in beta:
        nic = beta.index[beta.index != 'Intercept']
        # Descaling intercept
        beta['Intercept'] = beta['Intercept'] - np.sum(beta[nic] *
                                                       x['center'][nic] /
                                                       x['scale'][nic])
        # Descaling all other regression coefficients
        beta[nic] = beta[nic] / x['scale'][nic]

        log.debug('Regression coefficients destandardized (with Intercept).')
    else:
        beta = beta / x['scale']
        log.debug('Regression coefficients destandardized (no Intercept).')

    return beta
