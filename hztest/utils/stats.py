from typing import Optional

import numpy as np


def sample_event_times_in_interval(annual_occurrence_rate: float,
                                   interval_length: float,
                                   t0: float = 0.,
                                   rand_seed: Optional[int] = None
                                   ) -> np.ndarray:
    """

    """

    if rand_seed is not None:
        np.random.seed(rand_seed)

    n_events = np.random.poisson(annual_occurrence_rate * interval_length)

    event_times = np.random.uniform(low=t0,
                                    high=t0 + interval_length,
                                    size=n_events)
    return event_times


def poisson_likelihood(rate: float,
                       num_events: int,
                       time_interval: float = 1.,
                       not_modeled_val: float = 0.) -> float:
    if rate == 0:
        return poisson_likelihood_zero_rate(num_events, not_modeled_val)
    else:
        rt = rate * time_interval
        return np.exp(-rt) * rt**num_events / np.math.factorial(num_events)


def poisson_likelihood_zero_rate(num_events: int,
                                 not_modeled_val: float = 0.) -> float:
    if num_events == 0:
        return 1.
    elif num_events > 0:
        return not_modeled_val
    else:
        raise ValueError("num_events should be zero or a positive integer.")


def poisson_log_likelihood():
    raise NotImplementedError
