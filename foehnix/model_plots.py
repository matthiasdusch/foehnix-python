import numpy as np
import matplotlib.pyplot as plt

# TODO: think of a nice API
#       - either foehnix.plot.plot(fmm, which = 'loglik', 'timeseries',
#                                  'windrose', 'hovmoeller'...)
#       -or : model.plot(which='loglike', 'timeseries', 'windrose', ...)


def loglik(fmm, log=True, **kwargs):
    """
    Plots the log-likelihood sum path through the iterations of the EM algorithm

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    log : bool
        If True (default) the x-axis is shown on the log scale, else on
        the iteration scale
    """

    ll = fmm.optimizer['loglikpath'].copy()
    ylim = [ll.values.min(), ll.values.max()] + np.array([-0.05, 0.2]) *\
        (ll.values.max() - ll.values.min())
    xlabel = 'EM iteration'
    if log is True:
        ll.index = np.log(ll.index)
        xlabel = 'log(%s)' % xlabel

    fig, ax = plt.subplots(figsize=(10, 5))

    ll.plot(ax=ax, color=['0.5', '0.5', '0'], style=['--o', ':o', '-o'],
            markeredgecolor='C2', markerfacecolor='None')

    ax.get_lines()[-1].set_lw(2)
    ax.set(xlabel=xlabel, ylabel='log-likelihood', ylim=ylim,
           title='foehnix log-likelihood path')
    ax.legend(loc=1, ncol=3)
    fig.tight_layout()


def loglikcontribution(fmm, log=True, **kwargs):
    """
    Plots the log-likelihood with respect to initial log-likelihood

    Increasing values indicate positive log-likelihood contributions
    (improvement of the model).

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    log : bool
        If True (default) the x-axis is shown on the log scale, else on
        the iteration scale
    """

    ll = fmm.optimizer['loglikpath'].copy()
    ll = ll-ll.iloc[0]
    ylim = [ll.values.min(), ll.values.max()] + np.array([-0.05, 0.2]) * \
           (ll.values.max() - ll.values.min())
    xlabel = 'EM iteration'
    if log is True:
        ll.index = np.log(ll.index)
        xlabel = 'log(%s)' % xlabel

    fig, ax = plt.subplots(figsize=(10, 5))
    ll.plot(ax=ax, color=['0.5', '0.5', '0'], style=['--o', ':o', '-o'],
            markeredgecolor='C2', markerfacecolor='None')
    ax.get_lines()[-1].set_lw(2)
    ax.set(xlabel=xlabel, ylabel='log-likelihood contribution', ylim=ylim,
           title='foehnix log-likelihood contribution')
    ax.legend(loc=1, ncol=3)
    fig.tight_layout()


def coef(fmm, log=True, **kwargs):
    """
    Plots the estimated coefficients

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    log : bool
        If True (default) the x-axis is shown on the log scale, else on
        the iteration scale
    """

    path = fmm.optimizer['coefpath'].copy()

    xlabel = 'EM iteration'
    if log is True:
        path.index = np.log(path.index)
        xlabel = 'log(%s)' % xlabel

    if fmm.optimizer['ccmodel'] is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        comps = path.columns
    else:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        comps = path.columns.drop(fmm.coef['concomitants'].keys())
        conc = path.loc[:, fmm.coef['concomitants'].keys()]
    comps = comps.sort_values()

    # components
    ylim = ([path.loc[:, comps].values.min(), path.loc[:, comps].values.max()]
            + np.array([-0.05, 0.15]) * (path.loc[:, comps].values.max() -
                                         path.loc[:, comps].values.min()))
    path.loc[:, comps].plot(ax=ax, color=['xkcd:coral', 'xkcd:lightblue',
                                          'xkcd:red', 'xkcd:azure'],
                            style=['--', '--', '-', '-'])
    ax.set(xlabel=xlabel, ylabel='coefficient (components)', ylim=ylim,
           title='coefficient path (components)')
    ax.legend(loc=1, ncol=4)

    if fmm.optimizer['ccmodel'] is not None:

        ylim = [conc.values.min(), conc.values.max()] +\
               np.array([-0.05, 0.15]) * (conc.values.max() -
                                          conc.values.min())
        conc.plot(ax=ax1, color=['k', 'xkcd:coral'], style=['-', '--'])
        ax1.set(xlabel=xlabel, title='coefficient path (components)',
                ylabel='concomitant coefficients (standardized)', ylim=ylim)
        ax1.legend(loc=1, ncol=4)

    fig.tight_layout()

    # TODO conditional histogram plot