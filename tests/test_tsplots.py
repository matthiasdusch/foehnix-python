import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from foehnix.analysis_lots import tsplot, TSControl


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

