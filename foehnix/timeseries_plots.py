import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
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


def image(fmm, fun='freq', deltat=None, deltad=7,
          cmap=cm.get_cmap('Greys', 20), contours=False, contour_color='k',
          contour_levels=10, **kwargs):
    """
    foehnix Image Plot - Hovmoeller Diagram

    This plots a Hovmoeller Diagram with aggregated days on the x-axis and
    aggregated time on the y-axis.

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    fun : str or custom function
        Determines how to aggregate the Foehn probability
        Possible strings are:

        - ``'freq': frequency of foehn occurrence (probability >= 0.5)
        - ``'mean': mean probability
        - ``'occ': absolute occurrence of foehn (probabiliy >= 0.5)
        - ``'occ': absolute occurrence of no foehn (probabiliy < 0.5)
    deltat : int
        interval in seconds for time aggregation. Has to be a fraction of
        86400 (24h in seconds). If ``None`` (default) the interval of the time
        series will be used.
    deltad : int
        interval in days for daily aggregation. Default is 7.
    cmap : :py:class:`matplotlib.colormap`
        colormap to use for the Hovmoeller Diagram
    contours : Bool
        If ``True`` additional contour lines will be ploted. Default ``False``.
    contour_color : str
        Color of the contour lines. Default 'k' for black.
    contour_levels : float or sequence of floats
        Default ``10`` will plot 10 contour levels. If a sequence is provided
        contour levels will be plotted at the sequence values.
    kwargs :
        Possible keyword arguments for the figure:

        - ``'title'
        - ``'ylabel'
        - ``'xlabel'
    """

    if not isinstance(fmm, foehnix.Foehnix):
        raise AttributeError('First Attribute must be a foehnix mixture model '
                             ' instance.')
    # copy foehnix probability for easy access
    x = fmm.prob.prob.copy()

    # make sure index is a DatetimeIndex
    x.index = pd.to_datetime(x.index)
    mindt = x.index.to_series().diff().min().seconds
    # check if regular
    if not x.index.is_monotonic_increasing:
        raise RuntimeError('DataFrame index is not monotonic increasing!')
    # check if data is strictly increasing
    if not (mindt == x.index.to_series().diff().max().seconds):
        raise RuntimeError('DataFrame index is not strictly regular '
                           'increasing. This should be the case if a standard '
                           'Foehnix Mixture Model object is provided to the '
                           'function. If you provide a custom object, make '
                           'sure the index is strictly regular increasing!')

    # check deltat
    if deltat is None:
        deltat = mindt
        # TODO: das muss nicht aufgehen (%==0)! Bessere loesung ueberlegen!
    elif isinstance(deltat, int) and (86400 % deltat == 0):
        pass
    else:
        raise ValueError('`deltat` must be a fraction of 86400 seconds '
                         ' (1 day), provided as integer value.')
    if deltat < mindt:
        log.warning('Upsampling is not allowed: `deltat` must be greater or '
                    'equal to the time step of the DataFrame! Will use the'
                    'time step of the DataFrame instead.')
        # TODO: das muss nicht aufgehen (%==0)! Bessere loesung ueberlegen!
        deltat = mindt

    # check deltad
    if (not isinstance(deltad, int)) or (deltad > 365) or (deltad < 1):
        raise ValueError('`deltad` must be a integer within [1, 365]!')

    # checking colors
    # TODO

    # Aggregation function
    if fun == 'freq':
        def fun(fx): return (fx.dropna() >= 0.5).sum()/fx.notna().sum()
    elif fun == 'occ':
        def fun(fx): return (fx.dropna() >= 0.5).sum()
    elif fun == 'noocc':
        def fun(fx): return (fx.dropna() < 0.5).sum()
    elif fun == 'mean':
        def fun(fx): return fx.dropna().mean()
    elif callable(fun):
        log.info('Using provided aggregaton function %s' % fun.__name__)
        pass
    else:
        raise ValueError('Aggregation function `fun` must either be one of '
                         '"freq", "occ", "noocc" or "mean". Or a suitable '
                         'self provided function.')

    #
    # ----------- Regroup data -------------
    # make dataframe
    x = x.to_frame()
    # store day_of_year for deltad grouping
    x['doy'] = pd.TimedeltaIndex(x.index.dayofyear, 'd')

    # store seconds of day for deltat grouping
    secs = (x.index.values - x.index.values.astype('datetime64[D]')) /\
        np.timedelta64(1, 's')
    # 0 is end of day
    secs[secs == 0] = 3600*24
    # make secs a Timedeltaindex in order to use it as group frequency
    x['sod'] = pd.TimedeltaIndex(secs, 's')

    # TODO reto laesst den 29 Feber drinne und schneidet den Plot hinten ab
    # correct for leap years
    # x.loc[x.index.is_leap_year & (x['doy'] > 59), 'doy'] -= 1

    # 2. group by day and time
    grouper1 = pd.Grouper(key='doy', freq='%dd' % deltad)
    grouper2 = pd.Grouper(key='sod', freq='%ds' % deltat)
    x = x.groupby([grouper1, grouper2]).agg(lambda xx: fun(xx))
    x.index = x.index.remove_unused_levels()

    # ---------- plot stuff
    lenx = len(x.index.levels[0])
    leny = len(x.index.levels[1])
    Z = np.zeros((leny, lenx))

    fig, (ax, cax) = plt.subplots(1, 2, figsize=(12, 6.5),
                                  gridspec_kw={'width_ratios': [1, 0.03]})
    for nr, day in enumerate(x.index.levels[0]):
        Z[:, nr] = x.loc[day].values.squeeze()

    im = ax.imshow(Z, origin='lower', cmap=cmap, aspect='auto')

    # --- colorbar
    cbar = fig.colorbar(im, cax=cax)

    # --- contours
    if contours is True:
        # make cyclic boundaries
        zcon = np.zeros((Z.shape[0]+2, Z.shape[1]+2))

        # center
        zcon[1:-1, 1:-1] = Z

        # lower
        zcon[0, 1:-1] = Z[-1, :]
        zcon[0, 0] = Z[-1, -1]

        # upper
        zcon[-1, 1:-1] = Z[0, :]
        zcon[-1, -1] = Z[0, 0]

        # left
        zcon[1:-1, 0] = Z[:, -1]
        zcon[-1, 0] = Z[0, -1]

        # right
        zcon[1:-1, -1] = Z[:, 0]
        zcon[0, -1] = Z[-1, 0]
        # TODO: test larger border around the actual data (e.g. 2-5 point)

        """
        # lower
        zcon[0, 2:] = Z[-1, :]
        zcon[0, 1] = Z[-1, -1]
        zcon[0, 0] = Z[-1, -2]

        # upper
        zcon[-1, :-2] = Z[0, :]
        zcon[-1, -2] = Z[0, 0]
        zcon[-1, -1] = Z[0, 1]

        # left
        zcon[1:-1, 0] = Z[:, -1]

        # right
        zcon[1:-1, -1] = Z[:, 0]


        """
        xcon, ycon = np.meshgrid(np.arange(zcon.shape[1])-1,
                                 np.arange(zcon.shape[0])-1)

        ax.contour(xcon, ycon, zcon, colors=contour_color,
                   levels=contour_levels)

    # ---- kwargs for plotting
    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = 'foehnix Hovmoeller Diagram'

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    else:
        xlabel = 'time of the year'

    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    else:
        ylabel = 'time of the day'

    # --- x axis:
    yrindex = pd.DatetimeIndex(start='1900', end='1901', freq='d')
    month = [yr[:3] for yr in yrindex.month_name().unique()]
    yrindex = yrindex[yrindex.day == 1].dayofyear[:-1] - 1
    xt = np.arange(-0.5, lenx, 1/deltad)[yrindex]
    xt_minor = xt + np.append(np.diff(xt)/2, np.diff(xt[:2]/2))
    ax.set_xticks(xt)
    ax.set_xticklabels('', minor=False)
    ax.set_xticks(xt_minor, 'minor')
    ax.set_xticklabels(month, minor=True)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlim(-0.5, lenx-1.5)
    ax.set_xlabel(xlabel)

    # --- y axis:
    ax.set_yticks(np.arange(-0.5, leny, 3600/deltat))
    hindex = pd.DatetimeIndex(start='1900', end='1900-01-02', freq='h')
    ax.set_yticklabels(hindex.strftime('%H:%M'))
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.5, leny-0.5)

    # --- overall figure
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
