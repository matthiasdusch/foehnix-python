import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import logging

import foehnix


# logger
log = logging.getLogger(__name__)


class TSControl:
    """
    Control object for the foehnix time series plot function.

    Can be used to set variable names, colors and labels.
    """

    def __init__(self, fmm, userdict=None, **kwargs):
        """
        Initializes a control object instance.

        Parameters
        ----------
        fmm : :py:class:`foehnix.Foehnix`
            A foehnix mixture model object

        userdict : dict
            If provided will be used to overrule the default settings.\n
            If a key is not provided, default values will be used.
            Each entry consists of ``'key': ['varname', 'color', 'label']``.\n
            ``'varname'`` must be present in the DataFrame provided to the
            model.\n
            ``'color'`` must be any valid matplotlib-colorstring\n
            Possible keys and default values are:

            - ``'t': ['t', 'C3', 'air temperature [C]']``
            - ``'rh': ['rh', 'C2', 'relative humidity [%]']``
            - ``'diff_t': ['diff_t', 'C8', 'temperature difference [C]']``
            - ``'dd': ['dd', 'black', 'wind direction [deg]']``
            - ``'ff': ['ff', 'C0', 'wind speed [m/s]']``
            - ``'ffx': ['ffx', 'C4', 'gust speed [m/s]']``
            - ``'prob': [None, 'C6', 'probability']``

        kwargs :
            Can be used to quickly change the name of parameters within the
            Dataframe. E.g. ``tsc = TSControl(fmm, ff='windspd')`` will
            initialize a default control but change the column name from
            `ff` to `windspd`. These keyword arguments can also directly passed
            to the main plotting function
            :py:func:`foehnix.timeseries_plots.tsplot`. E.g.:
            ``tsplot(fmm, dd='winddir')``.
        """

        ctrldict = {
            't': ['t', 'C3', 'air temperature [C]'],
            'rh': ['rh', 'C2', 'relative humidity [%]'],
            'diff_t': ['diff_t', 'C1', 'temperature\n difference [C]'],
            'dd': ['dd', 'black', 'wind direction [deg]'],
            'ff': ['ff', 'C0', 'wind speed [m/s]'],
            'ffx': ['ffx', 'C4', 'gust speed [m/s]'],
            'prob': [None, 'C6', 'probability']
        }

        if userdict is None:
            userdict = {}

        for key in userdict.keys():
            try:
                ctrldict[key] = userdict[key]
            except KeyError:
                log.critical('KeyError %s not valid. Default values will be '
                             'used instead.')

        for key in kwargs.keys():
            try:
                ctrldict[key][0] = kwargs[key]
            except KeyError:
                log.critical('KeyError %s not valid. Default values will be '
                             'used instead.')

        # create a dict deciding which variables to plot
        doplot = {'0_temp': [],
                  '1_dT': [],
                  '2_wind': [],
                  '3_prob': []}

        if ctrldict['t'][0] in fmm.data:
            doplot['0_temp'].append('t')
        if ctrldict['rh'][0] in fmm.data:
            doplot['0_temp'].append('rh')

        if ctrldict['diff_t'][0] in fmm.data:
            doplot['1_dT'].append('diff_t')

        if ctrldict['dd'][0] in fmm.data:
            doplot['2_wind'].append('dd')
        if ctrldict['ff'][0] in fmm.data:
            doplot['2_wind'].append('ff')
        if ctrldict['ffx'][0] in fmm.data:
            doplot['2_wind'].append('ffx')

        if 'prob' in fmm.prob:
            doplot['3_prob'].append('prob')

        # remove empty values
        doplot = {k: v for k, v in doplot.items() if v != []}

        self.subplots = len(doplot)
        if self.subplots == 0:
            raise ValueError('Cannot find any of the required variables! One '
                             'reason: your variable names do not match any of '
                             'the default variable names. Default names can '
                             'changed with passing a userdict or '
                             'keyword arguments like ``ff="windspd"``. See '
                             ':py:class:`foehnix.timeseries_plots.TSControl` '
                             'for further details.')

        self.ctrldict = ctrldict
        self.doplot = doplot


def tsplot(fmm, start=None, end=None, ndays=10, tscontrol=None, show_n_plots=3,
           userdict=None, **kwargs):
    """
    Time series plot for foehnix models

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    start : str or Datetime timestamp
        First day to plot. Will be converted to a timestamp, if possible.
        Must be within the daterange of fmm.data.
    end : str or Datetime timestamp
        Last day to fully plot. Will be converted to a timestamp, if possible.
        Must be within the daterange of fmm.data.
    ndays : int
        Number of days which one plot shows. Default 10.
    tscontrol : :py:class:`foehnix.timeseries_plots.TSControl` object
        Can be predefined before calling this function or else will be
        initialized if None (default).
    show_n_plots : int
        How many figures will be opened, default is 1. After closing all, the
        next ones will be opened.
    userdict : dict
        alternative plotting parameters (variable names, colors, labels) which
        will be passed to TSControl. See
        :py:class:`foehnix.timeseries_plots.TSControl` for more details.
        If a suitable ``tscontrol`` argument is provided this dict will be
        ignored.
    kwargs
        will be passed to TSControl and can be used to rename the variable
        names of the ``fmm.data`` DataFrame. See
        :py:class:`foehnix.timeseries_plots.TSControl` for more details.
        If a suitable ``tscontrol`` argument is provided these argumetns will
        be ignored.
    """

    # If no control object is provided: Use default one.
    if not isinstance(tscontrol, TSControl):
        tscontrol = TSControl(fmm, userdict=userdict, **kwargs)

    if not isinstance(fmm, foehnix.Foehnix):
        raise AttributeError('First Attribute must be a foehnix mixture model '
                             ' instance.')

    # difference between time steps
    dt = fmm.data.index.to_series().diff().min()

    # check start and end values
    if start is not None:
        try:
            start = pd.to_datetime(start)
        except ValueError:
            log.critical('Could not convert start value to Datetime. Using '
                         'first data point instead.')
            start = fmm.data.index[0]
    else:
        start = fmm.data.index[0]

    if end is not None:
        try:
            end = pd.to_datetime(end)
        except ValueError:
            log.critical('Could not convert end value to Datetime. Using '
                         'last data point instead.')
            end = fmm.data.index[-1]
    else:
        end = fmm.data.index[-1]

    # starting dates to plot:
    # First date is 0UTC of the starting date
    # Then intervalls of ndays
    # Last plot will contain end date, but still be ndays long
    dates = pd.date_range(start.date(),
                          (end + pd.to_timedelta('23H')).date(),
                          freq='%dD' % ndays)

    # make sorted list of subplot keys
    dokeys = list(tscontrol.doplot.keys())
    dokeys.sort()

    # font sizes
    plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    for nr, i in enumerate(dates):

        # check if all plotdates out of DataFrame index
        if (i > fmm.data.index[-1]) or (i+1 < fmm.data.index[0]):
            log.critical('No data for timeseries plot between %s and %s. '
                         'Skipping.' % (i, i+1))
            continue

        fig, axs = plt.subplots(nrows=tscontrol.subplots, ncols=1,
                                figsize=(12, 2*tscontrol.subplots))

        for j in np.arange(tscontrol.subplots):

            vals = tscontrol.doplot[dokeys[j]]

            if 't' in vals:
                axs[j].plot(fmm.data.loc[i:i+1,
                            tscontrol.ctrldict['t'][0]],
                            color=tscontrol.ctrldict['t'][1])
                ylim = (fmm.data.loc[i:i+1, tscontrol.ctrldict['t'][0]].
                        min()-0.2,
                        fmm.data.loc[i:i+1, tscontrol.ctrldict['t'][0]].
                        max()+0.2)
                if np.isnan(ylim).any():
                    ylim = (0, 1)
                axs[j].set(ylabel=tscontrol.ctrldict['t'][2], ylim=ylim)
                axs[j].set_zorder(3)
                axs[j].patch.set_visible(False)

            if 'diff_t' in vals:
                axs[j].plot(fmm.data.loc[i:i+1,
                            tscontrol.ctrldict['diff_t'][0]],
                            color=tscontrol.ctrldict['diff_t'][1])
                ylim = (fmm.data.loc[i:i+1, tscontrol.ctrldict['diff_t'][0]].
                        min()-0.2,
                        fmm.data.loc[i:i+1, tscontrol.ctrldict['diff_t'][0]].
                        max()+0.2)
                if np.isnan(ylim).any():
                    ylim = (0, 1)
                axs[j].set(ylabel=tscontrol.ctrldict['diff_t'][2], ylim=ylim)

            if 'rh' in vals:
                axrh = axs[j].twinx()

                axrh.plot(fmm.data.loc[i:i+1, tscontrol.ctrldict['rh'][0]],
                          color=tscontrol.ctrldict['rh'][1])
                axrh.fill_between(fmm.data.loc[i:i+1].index,
                                  fmm.data.loc[i:i+1,
                                  tscontrol.ctrldict['rh'][0]],
                                  facecolor=tscontrol.ctrldict['rh'][1],
                                  alpha=0.3)
                axrh.set(ylabel=tscontrol.ctrldict['rh'][2],
                         ylim=(0, 140))
                axrh.set_yticks([20, 40, 60, 80])

            if 'ff' in vals:
                axff = axs[j].twinx()

                axff.plot(fmm.data.loc[i:i+1, tscontrol.ctrldict['ff'][0]],
                          color=tscontrol.ctrldict['ff'][1], zorder=1,
                          label='')
                axff.fill_between(fmm.data.loc[i:i+1].index,
                                  fmm.data.loc[i:i+1,
                                  tscontrol.ctrldict['ff'][0]],
                                  facecolor=tscontrol.ctrldict['ff'][1],
                                  alpha=0.3, zorder=1)
                ylim = (0, fmm.data.loc[i:i+1, tscontrol.ctrldict['ff'][0]].
                        max()+0.5)
                if np.isnan(ylim).any():
                    ylim = (0, 1)
                axs[j].set(ylabel=tscontrol.ctrldict['ff'][2], ylim=ylim)

            if 'ffx' in vals:
                if 'ff' not in vals:
                    axff = axs[j].twinx()

                axff.plot(fmm.data.loc[i:i+1, tscontrol.ctrldict['ffx'][0]],
                          color=tscontrol.ctrldict['ffx'][1], zorder=2,
                          label=tscontrol.ctrldict['ffx'][2])
                ylim = (0, fmm.data.loc[i:i+1, tscontrol.ctrldict['ffx'][0]].
                        max()+0.5)
                if np.isnan(ylim).any():
                    ylim = (0, 1)
                axs[j].set(ylim=ylim)

                if 'ff' not in vals:
                    axff.set(ylabel=tscontrol.ctrldict['ffx'][2])
                else:
                    lffx = axff.legend(loc=4)
                    lffx.set_zorder(100)

            if 'dd' in vals:
                axs[j].plot(fmm.data.loc[i:i+1, tscontrol.ctrldict['dd'][0]],
                            '.',
                            color=tscontrol.ctrldict['dd'][1], zorder=50)
                axs[j].set(ylabel=tscontrol.ctrldict['dd'][2],
                           ylim=(0, 360))

                axs[j].set_yticks([90, 180, 270])
                axs[j].set_zorder(3)
                axs[j].patch.set_visible(False)

            if 'prob' in vals:
                axs[j].plot(fmm.prob.loc[i:i+1, 'prob']*100,
                            color=tscontrol.ctrldict['prob'][1], zorder=1)
                axs[j].fill_between(fmm.prob.loc[i:i+1].index,
                                    fmm.prob.loc[i:i+1, 'prob']*100,
                                    facecolor=tscontrol.ctrldict['prob'][1],
                                    alpha=0.5, zorder=1)
                axs[j].set(ylabel=tscontrol.ctrldict['prob'][2],
                           ylim=(-2, 102))

            axs[j].set(xlim=(i, i+1))
            axs[j].grid(True, color='C7', linestyle=':')
            axs[j].grid(True, which='major', axis='x', linestyle='-')

            # xticks all axes
            xticks = int(max(1, np.round(ndays/10)))
            dloc = mdates.DayLocator(interval=xticks)
            if ndays <= 5:
                hloc = mdates.HourLocator(interval=6)
                axs[j].xaxis.set_minor_locator(hloc)
            axs[j].xaxis.set_major_locator(dloc)

        # xtickslabels, last axes only
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
        if ndays <= 5:
            axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

        # mask nan periods
        inan = fmm.prob.loc[i:i+1].index[np.where(fmm.prob.loc[i:i+1,
                                                  'prob'].isna())]
        if len(inan) > 0:
            axs[-1].plot(inan, np.zeros(inan.shape)-2, 'b+', ms=10,
                         color='C7',
                         clip_on=False, zorder=100)
        # prob > 50%
        i50 = fmm.prob.loc[i:i+1].index[np.where(fmm.prob.loc[i:i+1,
                                                 'prob'] > 0.5)]
        if len(i50) > 0:
            axs[-1].plot(i50, np.zeros(i50.shape)-2, 'b+', ms=10,
                         color=tscontrol.ctrldict['prob'][1],
                         clip_on=False, zorder=100)

            # plot grey boxes
            for ax in axs:
                ylim = ax.get_ylim()
                # fake timeseries, with dt/2 spacing
                p50 = pd.Series(index=pd.date_range(i-dt/2, i+1+dt/2,
                                                    freq=dt/2))
                p50.loc[i50] = ylim[1]
                p50.loc[i50-dt/2] = ylim[1]
                p50.loc[i50+dt/2] = ylim[1]
                ax.fill_between(p50.index, p50, y2=ylim[0],
                                facecolor='C7', alpha=0.25, zorder=50)

        fig.autofmt_xdate(rotation=0, ha='center')
        fig.suptitle('Foehn Diagnosis %s to %s' % (i.date(), (i+1).date()))
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        fig.subplots_adjust(hspace=0.02)

        if (nr+1) % show_n_plots == 0:
            plt.show()

    plt.show()
