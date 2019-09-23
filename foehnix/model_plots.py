import numpy as np
import matplotlib.pyplot as plt


def loglik(fmm, log=True):
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


def loglikcontribution(fmm, log=True):
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
    ax.set(xlabel=xlabel, ylabel='log-likelihood contribution', ylim=ylim,
           title='foehnix log-likelihood contribution')
    ax.legend(loc=1, ncol=3)
    fig.tight_layout()


def coef(fmm, log=True):
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
    path.loc[:, comps].plot(ax=ax, color=['xkcd:coral', 'xkcd:azure',
                                          'xkcd:red', 'xkcd:blue'],
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


def hist(fmm):
    """
    Conditional histogram plot

    Parameters
    ----------
    fmm : :py:class:`foehnix.Foehnix`
        A foehnix mixture model object
    """

    print('travis trace 2')
    # exclude missing values
    idx = fmm.prob.flag[fmm.prob.flag == 1].dropna().index
    # get y and probability
    y = fmm.data.loc[idx, fmm.predictor].copy()
    prob = fmm.prob.loc[idx, 'prob'].copy()

    # if censoring or truncation is used
    if np.isfinite(fmm.control.left):
        y = np.maximum(y, fmm.control.left)
    if np.isfinite(fmm.control.right):
        y = np.minimum(y, fmm.control.right)

    print('travis trace 3')
    print('len y = %d' % len(y))
    print('y = ', y)
    print('idx = ', idx)
    if len(y) > 0:
        print('travis trace 4')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        at = np.linspace(y.min(), y.max(), 501)
        bk = np.linspace(y.min(), y.max(), 50)

        hi1 = ax1.hist(y[prob < 0.5], bins=bk, density=True,
                       color='w', edgecolor='k')
        hi2 = ax2.hist(y[prob >= 0.5], bins=bk, density=True,
                       color='w', edgecolor='k')

        # density
        d1 = fmm.control.family.density(at, fmm.coef['mu1'],
                                        np.exp(fmm.coef['logsd1']))
        d2 = fmm.control.family.density(at, fmm.coef['mu2'],
                                        np.exp(fmm.coef['logsd2']))
        ax1.plot(at, d1, color='xkcd:red')
        ax2.plot(at, d2, color='xkcd:blue')

        ylim = (0, np.maximum(hi1[0].max(), hi2[0].max())+0.05)

        ax1.set(title='Conditional Histogram\nComponent 1 (no foehn)',
                ylim=ylim, xlim=(bk[0], bk[-1]), frame_on=False,
                xlabel=r'y[$\pi < 0.5$]', ylabel='Density')
        ax2.set(title='Conditional Histogram\nComponent 2 (foehn)',
                ylim=ylim, xlim=(bk[0], bk[-1]), frame_on=False,
                xlabel=r'y[$\pi < 0.5$]', ylabel='Density')

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
