import pytest
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from scipy.stats import logistic, norm
import os
import hashlib

from foehnix import demodata

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

