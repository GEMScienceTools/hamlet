import numpy as np
from hztest.utils.stats import poisson_likelihood, poisson_log_likelihood


def bin_observance_likelihood(num_events: int, bin_rate: float) -> float:
    not_modeled_val = 0.  # hardcoded in the RELM tests

    return poisson_likelihood(bin_rate, num_events, not_modeled_val)


def bin_observance_log_likelihood(num_events: int, bin_rate: float) -> float:
    if bin_rate == 0:
        return bin_observance_log_likelihood_zero_rate(num_events)
    else:
        return (-1 * bin_rate + num_events * np.log(bin_rate) -
                np.log(np.math.factorial(bin_rate)))


def bin_observance_log_likelihood_zero_rate(num_events: int) -> float:
    if num_events == 0:
        return np.log(1)
    elif num_events > 0:
        return -1 * np.infty
    else:
        raise ValueError


def model_log_likelihood(bin_event_counts, bin_rates):
    return np.sum(bin_observance_log_likelihood(bin_event_counts, bin_rates))
