"""
This module provides statistical functions that are not specific to any set of
tests.
"""
import math
from typing import Optional, Union, Tuple

import numpy as np

from math import gamma


def sample_num_events_in_interval(
    annual_occurrence_rate: float,
    interval_length: float,
    rand_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Returns the number of events in an interval, sampled from the
    Poisson distribution for the occurrence rate and interval length.
    """

    if rand_seed is not None:
        np.random.seed(rand_seed)

    n_events = np.random.poisson(annual_occurrence_rate * interval_length)

    return n_events


def sample_event_times_in_interval(
    annual_occurrence_rate: float,
    interval_length: float,
    t0: float = 0.0,
    rand_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Returns the times of events

    """

    n_events = sample_num_events_in_interval(
        annual_occurrence_rate, interval_length, rand_seed=rand_seed
    )

    event_times = np.random.uniform(
        low=t0, high=t0 + interval_length, size=n_events
    )
    return event_times


def sample_num_events_in_interval_array(
    annual_occurrence_rates: np.ndarray,
    interval_length: float,
    rand_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Returns the times of events from an array of event rates.
    """

    if rand_seed is not None:
        rng = np.random.default_rng(rand_seed)
    else:
        rng = np.random.default_rng()

    n_events = rng.poisson(annual_occurrence_rates * interval_length)

    return n_events


def sample_event_times_in_interval_array(
    annual_occurrence_rates: np.ndarray,
    interval_length: float,
    t0: float = 0.0,
    rand_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Returns the times of events from an array of event rates.
    """

    if rand_seed is not None:
        rng = np.random.default_rng(rand_seed)
    else:
        rng = np.random.default_rng()

    n_events = rng.poisson(annual_occurrence_rates * interval_length)

    total_n_events = sum(n_events)

    event_times = rng.uniform(
        low=t0, high=t0 + interval_length, size=total_n_events
    )

    event_times_list = []
    event_ind = 0
    for n in n_events:
        if n == 0:
            this_n_events = []
        else:
            this_n_events = event_times[event_ind : event_ind + n]
        event_times_list.append(this_n_events)
        event_ind += n

    return event_times_list


def poisson_likelihood(
    num_events: int,
    rate: float,
    time_interval: float = 1.0,
    not_modeled_val: float = 0.0,
) -> float:
    """
    Returns the Poisson likelihood of observing `num_events` in a
    `time_interval` given the `rate` of those events in the units of the time
    interval (i.e., if the `time_interval` is in years, the rate is the annual
    occurrence rate).

    If `rate` > 0, the Poisson likelihood is defined as

    :math:`L(n|rt) = (rt)^n \\exp(-rt) / n!`

    where `n` is `num_events`, `r` is `rate`, `t` is `time_interval`, and
    `L(n|rt)` is the Poisson likeihood.

    If `rate` = 0, then the `not_modeled_val` is returned.
    """
    if rate == 0:
        return poisson_likelihood_zero_rate(num_events, not_modeled_val)
    else:
        rt = rate * time_interval
        return np.exp(-rt) * rt ** num_events / np.math.factorial(num_events)


def poisson_likelihood_zero_rate(
    num_events: int, not_modeled_val: float = 0.0
) -> float:
    if num_events == 0:
        return 1.0
    elif num_events > 0:
        return not_modeled_val
    else:
        raise ValueError("num_events should be zero or a positive integer.")


def poisson_log_likelihood(
    num_events: int,
    rate: float,
    time_interval: float = 1.0,
    not_modeled_val: float = 0.0,
) -> float:

    if rate == 0.0:
        return np.log(poisson_likelihood_zero_rate(num_events, not_modeled_val))
    else:
        rt = rate * time_interval
        return (
            -1 * rt
            + num_events * math.log(rt)
            - math.log(np.math.factorial(num_events))
        )


def negative_binomial_distribution(
    num_events: int, prob_success: float, r_dispersion: Union[int, float]
) -> float:
    """
    Returns the negative binomial probability for observing

    """

    term_1 = (gamma(num_events + r_dispersion)) / (
        gamma(r_dispersion) * np.math.factorial(num_events)
    )

    term_2 = (1 - prob_success) ** r_dispersion

    term_3 = prob_success ** num_events

    return term_1 * term_2 * term_3


def estimate_negative_binom_parameters(
    samples, rate: Optional[float] = None, mle: bool = False
) -> Tuple[float, float]:
    if mle is False:
        mean = rate or np.mean(samples)
        variance = np.var(samples, ddof=1)
        prob_success = (variance - mean) / variance
        r_dispersion = mean ** 2 / (variance - mean)

    else:
        # maximum likelihood estimation will go here
        raise NotImplementedError

    return r_dispersion, prob_success


def kullback_leibler_divergence(p, q):
    """
    The Kullback-Leibler Divergence is a measure of the information loss in
    moving from a distribution P to a second distribution Q that may be a model
    or approximation.
    """

    # TODO: deal w/ zero probabilities

    pp = np.asarray(p)
    qq = np.asarray(q)

    return np.sum(pp * np.log(pp / qq))


def jensen_shannon_divergence(p, q):

    pp = np.asarray(p)
    qq = np.asarray(q)

    r = _mid_pt_measure(pp, qq)

    return 0.5 * (
        kullback_leibler_divergence(pp, r) + kullback_leibler_divergence(qq, r)
    )


def jensen_shannon_distance(p, q):
    return np.sqrt(jensen_shannon_divergence(p, q))


def _mid_pt_measure(p, q):
    return 0.5 * (p + q)
