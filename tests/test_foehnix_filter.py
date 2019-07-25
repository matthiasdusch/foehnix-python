import pytest
import numpy as np
import logging

from foehnix import foehnix_filter, filter_summary


def test_various_input(data, caplog):
    # test wrong data format
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(np.arange(100))
    assert e.match('x must be a pandas.DataFrame')

    # test wrong filter method
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(data, filter_method='foo')
    assert e.match('Filter method not understood')

    # test None filter method. This is no error!
    _ = foehnix_filter(data)
    assert 'No filter method specified' in caplog.records[-1].message

    # test wrong filter_method dict key
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(data, filter_method={'foo': [10, 20]})
    assert e.match('Filterdict key: foo not found in data')

    # test wrong filter_method dict value
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(data, filter_method={'dd': np.arange(10)})
    assert e.match('Not a valid value for Filterdict key: dd.')


def my_filter_fun_dataframe(x):
    filtered = np.zeros(len(x))
    filtered[np.where(x['dd'] % 5 == 0)] = 1
    filtered[np.isnan(x['dd'])] = np.nan
    return filtered


def my_filter_fun_single(x):
    filtered = np.zeros(len(x))
    filtered[np.where(x % 2 == 0)] = 1
    filtered[np.isnan(x)] = np.nan
    return filtered


def wrong_filter_fun_1(x):
    return np.arange(len(x))


def wrong_filter_fun_2(x):
    return np.arange(len(x)+1)


def test_filter_functions(data, caplog):
    caplog.set_level(logging.INFO)
    # test a filter function provided as filter_method working on data
    _ = foehnix_filter(data, filter_method=my_filter_fun_dataframe)
    assert ('Applied filter function my_filter_fun_dataframe' in
            caplog.records[-1].message)

    # test a filter function provided as dict_value provided on the key
    _ = foehnix_filter(data, filter_method={'dd': my_filter_fun_single})
    assert ('Applied filter function my_filter_fun_single' in
            caplog.records[-1].message)

    # test a filter function that does not return proper values (0, 1, NaN)
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(data, filter_method={'dd': wrong_filter_fun_1})
    assert e.match('The provided filter function must return a filtered')

    # test a filter function that return wrong sized filtered variable
    with pytest.raises(Exception) as e:
        _ = foehnix_filter(data, filter_method=wrong_filter_fun_2)
    assert e.match('size does not match the data frame')


def test_limit_filter_and_summary(data, caplog, capfd):
    caplog.set_level(logging.INFO)
    # test a standard limit filter
    ffo = foehnix_filter(data, filter_method={'dd': [90, 270]})
    assert ('Applied limit-filter [90.0 270.0] to key dd' in
            caplog.records[-1].message)

    # compare with manual result
    mandd = data['dd'].loc[(data['dd'] >= 90) & (data['dd'] <= 270)].index
    np.testing.assert_array_equal(ffo['good'], mandd)

    # test reverse limits
    ffo2 = foehnix_filter(data, filter_method={'dd': [270, 90]})
    assert ('Applied limit-filter [270.0 90.0] to key dd' in
            caplog.records[-1].message)

    # compare reverse limits with manual result
    mandd2 = data['dd'].loc[(data['dd'] <= 90) | (data['dd'] >= 270)].index
    np.testing.assert_array_equal(ffo2['good'], mandd2)

    # test summary
    filter_summary(ffo)
    out, err = capfd.readouterr()
    assert 'Total data set length: %20d' % len(data) in out
    assert 'The good (within filter): %17d' % len(mandd) in out
    assert ('The ugly (NaN, missing values): %11d' % np.isnan(data['dd']).sum()
            in out)


def test_multi_filter(data):
    # only use very high wind speeds for example
    ffo = foehnix_filter(data, filter_method={'dd': [10, 190],
                                              'ff': [25, 99]})

    # compare manual filter
    manf = data.loc[(data['dd'] >= 10) &
                    (data['dd'] <= 190) &
                    (data['ff'] >= 25)].index
    np.testing.assert_array_equal(ffo['good'], manf)
