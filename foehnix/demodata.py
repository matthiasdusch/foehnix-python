import os
import pandas as pd


def get_demodata(which='combined'):
    """
    Returns a demo data set from `Ellboegen` and `Sattelberg`.

    The foehnix package comes with two demo data sets containing
    observations from two automated weather stations in Tyrol,
    Austria. One station (Sattelberg) is located close to the main
    alpine ridge and is used as 'crest' station. The second station
    (Ellboegen) is located in the Wipp valley.
    The package demos and examples will estimate automated foehn
    classification models foehnix models for
    station Ellboegen using additional information from the
    crest station as concomitant variables and for custom wind
    filters foehnix_filter.

    Parameters
    ----------
    which : str
        Select one of the stations or a combined DataFrame:

        - ``'ellboegen'`` only observations from station Ellboegen
        - ``'sattelberg`` only observations from station Sattelberg
        - ``'combined'`` (default) observations from both stations.
          Suffix ``_crest`` indicates the mountain station Sattelberg.

    Returns
    -------
     : :py:class:`pandas.DataFrame`
        The selected data
    """

    wd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wd = os.path.join(wd, 'data/')

    ellboegen = pd.read_csv(os.path.join(wd, 'ellboegen.csv'), delimiter=';',
                            skipinitialspace=True)
    if which.lower() == 'ellboegen':
        return ellboegen

    sattelberg = pd.read_csv(os.path.join(wd, 'sattelberg.csv'), delimiter=';',
                             skipinitialspace=True)
    if which.lower() == 'sattelberg':
        return sattelberg

    data = pd.merge(ellboegen, sattelberg, on='timestamp', how='outer',
                    suffixes=('', '_crest'), sort=True)
    data.index = pd.to_datetime(data.timestamp, unit='s')

    data['diff_t'] = data['t_crest'] + 10.27 - data['t']

    if which.lower() == 'combined':
        return data
    else:
        raise ValueError('`which` must be either combined, ellboegen or '
                         'sattelberg.')
