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
    """Foehnix filter function
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
                    raise RuntimeError('Key: %s of filterdict not found in data'
                                       % key)

                tmp[x[key].isna(), nr] = np.nan

                # This filter will KEEP data between the two values
                if value[0] < value[1]:
                    tmp[np.where((x[key] >= value[0]) &
                                 (x[key] <= value[1])), nr] = 1

                else:
                    tmp[np.where((x[key] >= value[0]) |
                                 (x[key] <= value[1])), nr] = 1

                log.info('Applied limit-filter to key %s' % key)

        filtered = np.ones(len(x)) * np.nan
        filtered[np.all(tmp == 1, axis=1)] = 1
        filtered[np.any(tmp == 0, axis=1)] = 0

        # TODO Retos "tmp <- apply(tmp, 1, all)" macht entgegen der Doku
        # aus 0 und NAN -> FALSE und nicht NA. Kann man diskutieren.
        # test in R:
        # x <- cbind(x1 = c(1,0,NaN,1,NaN), x2=c(0,NaN,1,NaN,0))
        # print(x)
        # apply(x, 1, all)


    # 4. error
    else:
        raise RuntimeError('Filter method not understood')

    good = x.index[filtered == 1]
    bad = x.index[filtered == 0]
    ugly = x.index[np.isnan(filtered)]

    return {'good': good,
            'bad': bad,
            'ugly': ugly}
