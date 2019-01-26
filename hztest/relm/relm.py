import numpy as np

def bin_observance_likeihood(num_events, bin_rate):
    if bin_rate == 0:
        return bin_observance_likelihood_zero_rate(num_events)
    return bin_rate / np.math.factorial(num_events) * np.exp(bin_rate)


def bin_observance_log_likelihood(num_events, bin_rate):
    if bin_rate == 0:
        return bin_observance_log_likelihood_zero_rate(num_events)
    return (-1 * bin_rate + num_events * np.log(bin_rate) 
            - np.log(np.math.factorial(bin_rate)))


def bin_observance_likelihood_zero_rate(num_events):
    if num_events == 0:
        return 1
    elif num_events > 0:
        return 0
    else:
        raise ValueError


def bin_observance_log_likelihood_zero_rate(num_events):
    if num_events == 0:
        return np.log(1)
    elif num_events > 0:
        return -1 * np.infty
    else:
        raise ValueError


def model_log_likelihood(bin_event_counts, bin_rates):
    return np.sum(bin_observance_log_likelihood(bin_event_counts, bin_rates))