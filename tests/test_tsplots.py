import pytest
from unittest.mock import patch
import os
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from foehnix.analysis_plots import tsplot, TSControl, image


def test_tsplot_api_control(caplog, tyr_mod1):
    # Test some wrong inputs to the plot API
    with pytest.raises(AttributeError) as e:
        tsplot('foo')
    assert e.match('First Attribute must be a foehnix mixture model')

    # test control userdict and kwargs
    _tsc = TSControl(tyr_mod1, userdict={'foo': ['bar', 'r', 'foobar']})
    assert ('Key "foo" not valid. Default values will' in
            caplog.records[-1].message)
    _tsc = TSControl(tyr_mod1, foo='bar')
    assert ('Kwarg "foo" not valid. Default values will' in
            caplog.records[-1].message)

    # change variables to wrong names
    _tsc = TSControl(tyr_mod1, t='temp')
    assert ('Variable >> temp << not found in the data.' in
            caplog.records[-1].message)
    _tsc = TSControl(tyr_mod1, userdict={'ff': ['windsp', 0, 0],
                                         'ffx': ['gust', 0, 0],
                                         'dd': ['winddir', 0, 0],
                                         'rh': ['relhum', 0, 0],
                                         'diff_t': ['temp differ', 0, 0]})
    assert ('Variable >> relhum << not' in caplog.records[-5].message)
    assert ('Variable >> temp differ << not' in caplog.records[-4].message)
    assert ('Variable >> winddir << not' in caplog.records[-3].message)
    assert ('Variable >> windsp << not' in caplog.records[-2].message)
    assert ('Variable >> gust << not' in caplog.records[-1].message)


def test_tsplot_start_end(caplog, tyr_mod1):
    # test a out of bounds start date
    tyr_mod1.plot('timeseries', start='1. Feber 2005', end='2006-01-09',
                  showplot=False)
    assert ('Could not convert start value to Datetime.' in
            caplog.records[-1].message)

    # test a out of bounds end date
    tyr_mod1.plot('timeseries', start='2018-12-24', end='20022-02-22',
                  showplot=False)
    assert ('Could not convert end value to Datetime.' in
            caplog.records[-1].message)


@patch('matplotlib.pyplot.show')
def test_tsplot_show(_mock_show, caplog, tyr_mod1):
    # make 3 plots and show them without saving
    tyr_mod1.plot('timeseries', start='2011-10-01', end='2011-10-25', ndays=10,
                  showplot=True, saveplot=False, show_n_plots=1)
    # this should be the last figure
    assert ('Foehn Diagnosis 2011-10-21 to 2011-10-31' in
            plt.gcf().texts[0].get_text())


def test_tsplot_plot_and_save(tmp_path, caplog, tyr_mod1):
    # temporary file to store plots
    path = tmp_path / 'plots'
    path.mkdir()

    # make exactly 5 plots and save them
    tyr_mod1.plot('timeseries', start='2010-01-01', end='2010-01-22', ndays=5,
                  showplot=False, saveplot=True,
                  savedir=path.as_posix())
    assert os.path.exists(os.path.join(path.as_posix(),
                                       'foehnix_timeseries_04.png'))
    assert os.path.exists(os.path.join(path.as_posix(),
                                       'foehnix_timeseries_05.png')) is False

    # make 1 pdf plot
    tyr_mod1.plot('timeseries', start='2010-01-01', end='2010-01-02', ndays=3,
                  showplot=False, saveplot=True,
                  savedir=path.as_posix(), savefilename='tsplot.pdf')
    assert os.path.exists(os.path.join(path.as_posix(), 'tsplot_00.pdf'))


def test_image_api(tyr_mod1):
    # Test some wrong inputs to the plot API
    with pytest.raises(AttributeError) as e:
        image('foo')
    assert e.match('First Attribute must be a foehnix mixture model')
    # wrong function
    with pytest.raises(ValueError) as e:
        tyr_mod1.plot('image', fun='foo')
    assert e.match('Aggregation function `fun` must either be one')

    # modify the index to make it non monotonic
    fakemod = deepcopy(tyr_mod1)
    idx = fakemod.prob.prob.index.tolist()
    idx[1] = idx[3]
    fakemod.prob.prob.index = idx
    with pytest.raises(RuntimeError) as e:
        fakemod.plot('image')
    assert e.match('DataFrame index is not monotonic increasing!')

    # modify the index to make it non regular
    idx[1] = idx[2] - pd.to_timedelta('1S')
    fakemod.prob.prob.index = idx
    with pytest.raises(RuntimeError) as e:
        fakemod.plot('image')
    assert e.match('DataFrame index is not strictly regular')

    # give a wrong delta t
    with pytest.raises(ValueError) as e:
        tyr_mod1.plot('image', deltat=1000)
    assert e.match('`deltat` must be a fraction of 86400 seconds')

    # give a wrong delta d
    with pytest.raises(ValueError) as e:
        tyr_mod1.plot('image', deltad=400)
    assert e.match('must be a integer within')


def test_image_plot_and_save(tmp_path, caplog, tyr_mod1):
    # temporary file to store plots
    path = tmp_path / 'plots'
    path.mkdir()

    # basic hovmoeller plot, but a too small delta t should log a warning
    tyr_mod1.plot('image', deltat=1800, showplot=False, saveplot=True,
                  savedir=path.as_posix())
    # check for the upsampling message
    assert ('Upsampling is not allowed:' in caplog.records[-1].message)
    # check if plot got saved correctly
    assert os.path.exists(os.path.join(path.as_posix(),
                                       'foehnix_hovmoeller.png'))
    # test some labels
    assert 'Hovmoeller Diagram' in plt.gcf().axes[0].get_title()
    assert 'time of the year' in plt.gcf().axes[0].get_xlabel()
    assert 'time of the day' in plt.gcf().axes[0].get_ylabel()

    # advanced plot with custom title and pdf
    tyr_mod1.plot('image', fun='occ', deltat=3600, title='Test title',
                  xlabel='Test xlabel', ylabel='Test ylabel',
                  showplot=False, saveplot=True, savedir=path.as_posix(),
                  savefilename='foehnix_hov.pdf')
    # check if plot got saved correctly
    assert os.path.exists(os.path.join(path.as_posix(),
                                       'foehnix_hov.pdf'))
    assert 'Test title' in plt.gcf().axes[0].get_title()
    assert 'Test xlabel' in plt.gcf().axes[0].get_xlabel()
    assert 'Test ylabel' in plt.gcf().axes[0].get_ylabel()
