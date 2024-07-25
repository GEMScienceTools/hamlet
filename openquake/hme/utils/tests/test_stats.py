import unittest

import numpy as np
from scipy.stats import nbinom, poisson

from openquake.hme.utils.stats import (
    sample_num_events_in_interval,
    sample_event_times_in_interval,
    sample_num_events_in_interval_array,
    sample_event_times_in_interval_array,
    poisson_likelihood,
    poisson_likelihood_zero_rate,
    poisson_log_likelihood,
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
    kullback_leibler_divergence,
    jensen_shannon_distance,
    jensen_shannon_divergence,
    _mid_pt_measure,
    geom_mean,
    weighted_geom_mean,
)


def test_sample_num_events_in_interval():
    rate = 1e-3
    interval = 750.0

    expected = 1
    n_events = sample_num_events_in_interval(rate, interval, rand_seed=0)
    assert n_events == expected

    expected = 0
    n_events = sample_num_events_in_interval(rate, interval, rand_seed=1)
    assert n_events == expected


def test_sample_event_times_in_interval():
    rate = 0.3
    interval = 20

    event_times = sample_event_times_in_interval(rate, interval, rand_seed=0)
    expected = np.array(
        [
            11.36089122,
            18.51193277,
            1.42072116,
            1.74258599,
            0.40436795,
            16.65239691,
            15.56313502,
            17.40024296,
            19.57236684,
            15.98317128,
            9.22958725,
        ]
    )
    np.testing.assert_almost_equal(event_times, expected, decimal=5)


def test_sample_num_events_in_interval_array():
    rates = np.array([1e-3, 1e-2, 1e-1])
    interval = 750.0

    expected = [1, 2, 89]
    n_events = sample_num_events_in_interval_array(
        rates, interval, rand_seed=0
    )
    assert np.all(n_events == expected)

    expected = [2, 7, 76]
    n_events = sample_num_events_in_interval_array(
        rates, interval, rand_seed=1
    )
    assert np.all(n_events == expected)


def test_sample_event_times_in_interval_array():
    rates = np.array([0.3, 0.5, 0.7])
    interval = 20.0

    event_times = sample_event_times_in_interval_array(
        rates, interval, rand_seed=0
    )

    assert len(event_times) == len(rates)

    expected = [
        np.array([16.31707108, 0.05477, 17.14808553]),
        np.array(
            [
                0.67171151,
                14.59310893,
                3.51311241,
                17.26357845,
                10.8292244,
                5.99423781,
                8.45374442,
                0.56639342,
                2.48566553,
                13.41248829,
                12.94379023,
            ]
        ),
        np.array(
            [
                12.30770223,
                7.67355109,
                19.94419872,
                19.61670678,
                13.71083969,
                13.00918553,
                13.76893461,
                7.77842848,
                2.7019301,
                14.4297668,
                10.50708645,
                6.20483751,
                9.71670718,
                17.78975669,
            ]
        ),
    ]

    for i, et in enumerate(event_times):
        np.testing.assert_almost_equal(et, expected[i], decimal=5)


def test_poisson_likelihood():
    num_events = 2
    rate = 0.5
    interval = 1.0

    # nonzero rates, events
    expected = poisson.pmf(num_events, rate * interval)
    result = poisson_likelihood(num_events, rate, interval)
    np.testing.assert_almost_equal(result, expected, decimal=5)

    # zero events
    expected = poisson.pmf(0, rate * interval)
    result = poisson_likelihood(0, rate, interval, not_modeled_val=0.0)
    assert result == expected

    # zero rate (should send to poisson_likelihood_zero_rate)
    # default (zero) not_modeled_val
    # nonzero observed events
    rate_zero = 0.0
    not_modeled_val_default = 0.0
    expected = not_modeled_val_default
    result = poisson_likelihood(num_events, rate_zero, interval)
    assert result == expected

    # zero rate (should send to poisson_likelihood_zero_rate)
    # non_default not_modeled_val
    # nonzero observed events
    rate_zero = 0.0
    not_modeled_val = 0.5
    expected = not_modeled_val
    result = poisson_likelihood(
        num_events, rate_zero, interval, not_modeled_val=not_modeled_val
    )
    assert result == expected

    # zero rate (should send to poisson_likelihood_zero_rate)
    # non_default not_modeled_val
    # zero observed events
    rate_zero = 0.0
    not_modeled_val = 0.5
    expected = 1.0
    result = poisson_likelihood(0, rate_zero, interval)
    assert result == expected


def test_poisson_likelihood_zero_rate():
    num_events = 2
    interval = 1.0

    expected = 1.0
    result = poisson_likelihood_zero_rate(num_events, interval)
    assert result == expected

    expected_nm_val = 0.5
    result = poisson_likelihood_zero_rate(
        num_events, not_modeled_val=expected_nm_val
    )
    assert result == expected_nm_val


def test_geom_mean_many_vals():
    vals = [1, 2, 3, 4, 5]

    expected = 2.60517

    result = geom_mean(*vals)

    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_geom_mean_val_list():
    vals = [1, 2, 3, 4, 5]

    expected = 2.60517

    result = geom_mean(vals)

    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_weighted_geom_mean_many_vals():
    vals = [1, 2, 3, 4, 5]
    weights = [2, 5, 6, 4, 3]

    expected = 2.77748

    result = weighted_geom_mean(*vals, weights=weights)

    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_weighted_geom_mean_val_list():
    vals = [1, 2, 3, 4, 5]
    weights = [2, 5, 6, 4, 3]

    expected = 2.77748

    result = weighted_geom_mean(vals, weights=weights)

    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_weighted_geom_mean_no_weights():
    vals = [1, 2, 3, 4, 5]

    expected = 2.60517

    result = weighted_geom_mean(*vals, weights=None)

    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_weighted_geom_mean_val_list_no_weights():
    vals = [1, 2, 3, 4, 5]

    expected = 2.60517

    result = weighted_geom_mean(vals, weights=None)

    np.testing.assert_almost_equal(result, expected, decimal=4)
