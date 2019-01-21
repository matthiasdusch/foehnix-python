import numpy as np
import pandas as pd
import logging

# logger
log = logging.getLogger(__name__)


def _check_filter_function(idx, filtered):
    # helper function to check the results of a provided filter function
    # check if filtered is a pandas index, else make one
    if not isinstance(filtered, pd.Index):
        filtered = pd.Index(filtered)
    # check if all returned values exist in the original data frame
    if not filtered.isin(idx).all():
        raise RuntimeError('The provided filter function returned indices '
                           'which are not in the original DataFrame')
    return filtered


def foehnix_filter(x, filter_method=None):
    """
    Evaluates Data Filter Rules for foehnix Mixture Model Calls

    :py:class:`foehnix.Foehnix` models allow to specify an optional
    :py:func:`foehnix.foehnix_filter`. If a filter is given only a subset of
    the data set provided to :py:class:`foehnix.Foehnix` is used for the foehn
    classification.

    A typical example is a wind direction filter such that
    only observations (times) are used where the observed
    wind direction was within a user defined wind sector
    corresponding to the wind direction during foehn events
    for a specific location.

    However, the filter option allows to even implement complex
    filter rules if required. The 'Details' section contains
    further information and examples how this filter rules can be used.

    The most common filter rule: The filter is a `dict` where the dict-keys
    are column-names of the DataFrame `x`. The dict-values are lists of length
    2 and define the range which should be used to filter the data. Example:

    filter_method = {'dd': [43, 223]}

    This will keep all wind directions `dd` between 43 and 223 degrees

    The dict can contain several items to filter, e.g. to also limit the wind
    direction range at a crest station. Example:

    filter_method = {'dd': [43, 223], 'dd_crest': [90, 270]}

    Parameters
    ----------
    x : :py:class:`pandas.DataFrame`
        containing the observations
    filter_method : None, custom function or dict
        Can be one of the following:

        - `None`: No filter will be applied
        - `func`: A custom function which will be applied on ``x``
        - `dict`: Keys must be columns of ``x``, values can either be a custom
          function on ``x[key]`` or a list of length two.

    Returns
    -------
    dict
        A dictionary containing the following items:

        - `dict['good']`: all indices of ``x`` within the filter values
        - `dict['bad']` : all indices of ``x`` outside the filter values
        - `dict['ugly']`: all indices where one of the filter variables is NAN
    """
    # check x
    if not isinstance(x, pd.DataFrame):
        raise RuntimeError('x must be a pandas.DataFrame')

    # check filter_method
    # 1. None: return full index
    # 2. Function: Apply function to x, check the result and return if sensible
    # 3. dict, where keys are columns of x and items are values or functions

    # 1. None: return full index
    if filter_method is None:
        filtered = np.ones(len(x))

    # 2. Function: Apply function to x, check the result and return if sensible
    elif callable(filter_method):
        # TODO:
        raise RuntimeError('I have to rethink this part later...')
        filtered = filter_method(x)
        filtered = _check_filter_function(x.index, filtered)
        log.info('Applied filter function  %s' % filter_method.__name__)

    # 3. dict, where keys are columns of x and items are values or functions
    elif isinstance(filter_method, dict):
        # start with a matrix, with size of the dict, length of x and zeros
        tmp = np.zeros([len(x), len(filter_method)])

        # loop over dict and apply every filter
        for nr, (key, value) in enumerate(filter_method.items()):

            #  _x = x.loc[good].copy()

            if callable(value):
                # TODO:
                raise RuntimeError('I have to rethink this part later...')
                tmp = value(x.loc[good])
                good = _check_filter_function(good, tmp)
                log.info('Applied filter function  %s' % value.__name__)

            elif len(value) == 2:
                if key not in x.columns:
                    raise RuntimeError('Filterdict key: %s not found in data'
                                       % key)

                tmp[x[key].isna(), nr] = np.nan

                # This filter will KEEP data between the two values
                if value[0] < value[1]:
                    tmp[np.where((x[key] >= value[0]) &
                                 (x[key] <= value[1])), nr] = 1

                else:
                    tmp[np.where((x[key] >= value[0]) |
                                 (x[key] <= value[1])), nr] = 1

                log.info('Applied limit-filter [%.1f %.1f] to key %s' % (
                    value[0], value[1], key))

        # - If at least one element is NAN     -> set to NAN
        # - If all elements are TRUE (=1)      -> set to 1
        # - Else: One ore more are FALSE (=0)  -> set to 0
        filtered = np.zeros(len(x))
        filtered[np.any(np.isnan(tmp), axis=1)] = np.nan
        filtered[np.all(tmp == 1, axis=1)] = 1

    # 4. error
    else:
        raise RuntimeError('Filter method not understood')

    good = x.index[filtered == 1]
    bad = x.index[filtered == 0]
    ugly = x.index[np.isnan(filtered)]

    return {'good': good,
            'bad': bad,
            'ugly': ugly}
