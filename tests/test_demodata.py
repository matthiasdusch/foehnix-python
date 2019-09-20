import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
import pandas as pd
import os
import hashlib
import sys

from foehnix import get_demodata, Foehnix

# specify data directory
DDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDir = os.path.join(DDir, 'data/')

# nowindows = pytest.mark.skipif(sys.platform == 'win32',
#                               reason='md5 hashsums are calculated
#                                      'differently under Windows')
# usage: @nowindows
#        def foo():


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


def prob_to_csv(fmo, fpath):
    """
    Small helper function to write the Foehnix probabilites to a csv file

    The format is rather specific in order to compare hash values with the R
    version.

    Parameters
    ----------
    fmo : Foehnix class object
    fpath : PosixPath to the csv file
    """
    # Format the flag as Integer with 4 characters
    fmo.prob['flag'] = fmo.prob['flag']. \
        map(lambda x: '%5d' % x if not pd.isna(x) else '%5s' % 'NA')

    # Format the timestamp as Unix time
    fmo.prob.index = (fmo.prob.index.astype(np.int64)/1e9).astype(int)

    # Format the index and header label
    index_label = '%10s' % 'timestamp'
    header = fmo.prob.columns.tolist()
    header[0] = '%7s' % header[0]
    header[1] = '%5s' % header[1]

    # write probability output to file an check csv
    fpath.write_text(fmo.prob.to_csv(float_format='%7.3f', na_rep='%7s' % 'NA',
                                     sep=';', header=header,
                                     index_label=index_label))


def test_tyrol(tmpfile):
    # load data
    tyrol = get_demodata('tyrol')

    # specify wind filter
    tyr_filter = {'dd': [43, 223], 'dd_crest': [90, 270]}

    tyr1 = Foehnix('ff', tyrol, filter_method=tyr_filter, verbose=False)
    prob_to_csv(tyr1, tmpfile)
    readfile = hashlib.md5(tmpfile.open('rb').read())
    # Hashsum is in agreement with github.com/retostauffer/Rfoehnix
    assert readfile.hexdigest() == '43562856bc775e05f8f17d3deb27447d'

    tyr2 = Foehnix('ff', tyrol, concomitant=['rh', 'diff_t'],
                   filter_method=tyr_filter, verbose=False)
    prob_to_csv(tyr2, tmpfile)
    readfile = hashlib.md5(tmpfile.open('rb').read())
    # Hashsum is in agreement with github.com/retostauffer/Rfoehnix
    assert readfile.hexdigest() == 'f1cb14d57e0a8697fd8d0e3c6c0360ff'


def test_ellboegen(tmpfile):
    # load data
    ellboegen = get_demodata('ellboegen')

    # specify wind filter
    ellboegen_filter = {'dd': [43, 223]}

    ell1 = Foehnix('ff', ellboegen, filter_method=ellboegen_filter,
                   verbose=False)
    prob_to_csv(ell1, tmpfile)
    readfile = hashlib.md5(tmpfile.open('rb').read())
    # Hashsum is in agreement with github.com/retostauffer/Rfoehnix
    assert readfile.hexdigest() == 'b604d81ce93f47d6479acf57c52f53a2'

    ell2 = Foehnix('ff', ellboegen, concomitant=['rh'],
                   filter_method=ellboegen_filter, verbose=False)
    prob_to_csv(ell2, tmpfile)
    readfile = hashlib.md5(tmpfile.open('rb').read())
    # Hashsum is in agreement with github.com/retostauffer/Rfoehnix
    assert readfile.hexdigest() == 'c2fadc758ddbcb1973872f4a8c4f3db5'
