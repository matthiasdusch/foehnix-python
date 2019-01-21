.. currentmodule:: foehnix-python

.. _foehnix-demo:

Getting started
===============

.. ipython:: python
   :suppress:

    import numpy as np
    np.set_printoptions(threshold=10)

Foehnix and Pandas

.. ipython:: python

    import foehnix
    import pandas as pd

Load Ellboegen and Sattelberg data, merge and create filter

.. ipython:: python

    ellboegen = pd.read_csv('../data/ellboegen.csv', delimiter=';', skipinitialspace=True)
    sattelberg = pd.read_csv('../data/sattelberg.csv', delimiter=';', skipinitialspace=True)
    ellboegen.head()
    data = pd.merge(ellboegen, sattelberg, on='timestamp', how='outer', suffixes=('', '_crest'), sort=True)
    data.index = pd.to_datetime(data.timestamp, unit='s')
    train = data.iloc[:-10].copy()
    test = data.iloc[-10:].copy()
    train['diff_t'] = train['t_crest'] + 10.27 - train['t']
    ddfilter = {'dd': [43, 223], 'dd_crest': [90, 270]}

Run the model and show a summary

.. ipython:: python

    model = foehnix.Foehnix('diff_t', train, concomitant='ff', filter_method=ddfilter, switch=True, verbose=True)
    model.summary()

Plot some model assessments

.. ipython:: python

    @savefig loglike.png
    model.plot('loglik', log=False)
    @savefig loglikecontribution.png
    model.plot('loglikcontribution', log=True)
    @savefig coef.png
    model.plot('coef', log=True)
