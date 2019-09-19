import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from scipy.stats import logistic, norm
import os
import hashlib

from foehnix import get_demodata, Foehnix

# specify data directory
DDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDir = os.path.join(DDir, 'data/')


def test_md5():
    hashdict = {'ellboegen.csv': 'a48b770830a688cb31dd56b9900bd46a',
                'sattelberg.csv': '2cc74fcb6c02361f0fed26e18be247a2',
                'viejas.csv': 'f1a19ec04637fc231bc33b052ca2aeb5',
                'luckyfive.csv': '529ef6465418c45a7656c8dc2c027c73'}

    for fname, md5hash in hashdict.items():
        file = hashlib.md5(open(os.path.join(DDir, fname), 'rb').read())

        assert md5hash == file.hexdigest()


def test_get_demodata():
    # wrong input
    with pytest.raises(ValueError) as e:
        get_demodata('foo')
    assert e.match('must be either `tyrol`, `california`')

    # cali
    cali = get_demodata('california')
    viejas = get_demodata('viejas')
    lucky = get_demodata('luckyfive')
    npt.assert_array_equal(cali['air_temp'].dropna(), viejas['air_temp'])
    npt.assert_array_equal(cali['air_temp_crest'].dropna(), lucky['air_temp'])

    # tirol
    tirol = get_demodata('tyrol')
    ellboegen = get_demodata('ellboegen')
    sattelberg = get_demodata('sattelberg')
    npt.assert_array_equal(tirol['t'].dropna(), ellboegen['t'].dropna())
    npt.assert_array_equal(tirol['t_crest'].dropna(), sattelberg['t'].dropna())


def test_tyrol(tmpfile):
    # load data
    tyrol = get_demodata('tyrol')

    # specify wind filter
    tyr_filter = {'dd': [43, 223], 'dd_crest': [90, 270]}

    tyr1 = Foehnix('diff_t', tyrol, concomitant=['rh', 'ff'],
                   filter_method=tyr_filter, switch=True, verbose=False)

    tyr1


def test_ellboegen(tmpfile):
    # load data
    ellboegen = get_demodata('ellboegen')

    # specify wind filter
    ell_filter = {'dd': [43, 233]}

    ell1 = Foehnix('ff', ellboegen, filter_method=ell_filter, verbose=False)
    ell1
