import logging

import numpy as np
from openquake.hme.utils.stats import poisson_likelihood, poisson_log_likelihood


def bin_observance_likelihood(num_events: int, bin_rate: float) -> float:
    not_modeled_val = 0.0  # hardcoded in the RELM tests

    return poisson_likelihood(bin_rate, num_events, not_modeled_val)


def bin_observance_log_likelihood(num_events: int, bin_rate: float) -> float:
    """
    Calculates the log-likelihood of observing `num_events` in a bin given the
    rate of those events occuring in the observation time period, `bin_rate`.
    If the `bin_rate` = 0, and `num_events` is > 0, then negative infinity is
    returned.
    """

    if bin_rate == 0:
        return bin_observance_log_likelihood_zero_rate(num_events)
    else:
        return poisson_log_likelihood(num_events, bin_rate)


def bin_observance_log_likelihood_zero_rate(num_events: int) -> float:
    if num_events == 0:
        return np.log(1)
    elif num_events > 0:
        return -1 * np.infty
    else:
        raise ValueError


def model_log_likelihood(bin_event_counts, bin_rates):
    """
    Calculates the log-likelihood of a hazard model given the number of event
    counts in each bin (a bin is a magnitude bin inside of a spatial bin or
    cell) given the occurrence rates of earthquakes for that bin. 

    The log-likehood for the model is the sum of the log-likelihoods for each
    bin in the model (corresponding to the product of the likelihoods).
    """
    return np.sum(bin_observance_log_likelihood(bin_event_counts, bin_rates))
