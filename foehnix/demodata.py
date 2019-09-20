import os
import pandas as pd


def get_demodata(which):
    """
    Returns a demo data.

    The foehnix package comes with two sets of meteorological observations: one
    for Tyrol, Austria, and one for Southern California, USA.
    Both data sets come with observations from two stations, one valley station
    (or target station) and one station further upstream of the main foehn wind
    direction (crest station) used to filter the data (see foehnix_filter).
    For Tyrol, observations for station Ellbögen (valley) and station
    Sattelberg (crest) are included, the Californian data set consists of the
    crest station 'Lucky Five Ranch' and the valley station 'Viejas Casino and
    Resort'.

    Parameters
    ----------
    which : str
        Select one of the stations or a combined DataFrame:

        - ``'tyrol'`` returns the combined Tyrolian data set
          Suffix ``_crest`` indicates the mountain station Sattelberg.
        - ``'california'`` returns the combined California data set
          Suffix ``_crest`` indicates the mountain station Lucky Five Range.
        - ``'ellboegen'`` only observations from station Ellboegen
        - ``'sattelberg`` only observations from station Sattelberg
        - ``'luckyfive'`` only observations from station Lucky Five Range
        - ``'viejas'`` only observations from station Viejas Casino and Resort

    Returns
    -------
     : :py:class:`pandas.DataFrame`
        The selected data
    """

    wd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wd = os.path.join(wd, 'data/')

    if which.lower() in ['ellboegen', 'sattelberg', 'viejas', 'luckyfive']:
        data = pd.read_csv(os.path.join(wd, '%s.csv' % which.lower()),
                           delimiter=';', skipinitialspace=True)
        data.index = pd.to_datetime(data.timestamp, unit='s')
        return data

    elif which.lower() == 'tyrol':
        ellboegen = pd.read_csv(os.path.join(wd, 'ellboegen.csv'),
                                delimiter=';', skipinitialspace=True)
        sattelberg = pd.read_csv(os.path.join(wd, 'sattelberg.csv'),
                                 delimiter=';', skipinitialspace=True)

        data = pd.merge(ellboegen, sattelberg, on='timestamp', how='outer',
                        suffixes=('', '_crest'), sort=True)
        data.index = pd.to_datetime(data.timestamp, unit='s')

        data['diff_t'] = data['t_crest'] + 10.27 - data['t']

        return data

    elif which.lower() == 'california':
        viejas = pd.read_csv(os.path.join(wd, 'viejas.csv'),
                             delimiter=';', skipinitialspace=True)
        lucky = pd.read_csv(os.path.join(wd, 'luckyfive.csv'),
                            delimiter=';', skipinitialspace=True)

        data = pd.merge(viejas, lucky, on='timestamp', how='outer',
                        suffixes=('', '_crest'), sort=True)
        data.index = pd.to_datetime(data.timestamp, unit='s')

        data['diff_air_temp'] = (data['air_temp_crest'] + 7.30 -
                                 data['air_temp'])

        return data

    else:
        raise ValueError('`which` must be either `tyrol`, `california`, '
                         '`ellboegen`, `sattelberg`, `viejas` or `luckyfive`')
