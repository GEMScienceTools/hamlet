import numpy as np

def sample_event_times_in_interval(annual_occurrence_rate: float, 
                              interval_length: float, 
                              t0: float=0., rand_seed=None):


    if rand_seed is not None:
        np.random.seed(rand_seed)

    n_events = np.random.poisson(annual_occurrence_rate * interval_length)

    event_times = np.random.uniform(low=t0, high=t0+interval_length, 
                                    size=n_events)
    return event_times


def bin_observance_likelihood(num_events: int, bin_rate: float) -> float:
    if bin_rate == 0:
        return bin_observance_likelihood_zero_rate(num_events)
    else:
        return bin_rate / np.math.factorial(num_events) * np.exp(bin_rate)


def bin_observance_log_likelihood(num_events: int, bin_rate: float) -> float:
    if bin_rate == 0:
        return bin_observance_log_likelihood_zero_rate(num_events)
    else:
        return (-1 * bin_rate + num_events * np.log(bin_rate) 
                - np.log(np.math.factorial(bin_rate)))


def bin_observance_likelihood_zero_rate(num_events: int) -> float:
    if num_events == 0:
        return 1
    elif num_events > 0:
        return 0
    else:
        raise ValueError


def bin_observance_log_likelihood_zero_rate(num_events: int) -> float:
    if num_events == 0:
        return np.log(1)
    elif num_events > 0:
        return -1 * np.infty
    else:
        raise ValueError


def model_log_likelihood(bin_event_counts, bin_rates):
    return np.sum(bin_observance_log_likelihood(bin_event_counts, bin_rates))
