"""
Utility functions for running tests in the RELM model test framework.
"""
import logging
from typing import Sequence, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm
from scipy.stats import poisson, nbinom
from geopandas import GeoSeries, GeoDataFrame

from openquake.hme.utils.bins import SpacemagBin
from openquake.hme.utils import (
    get_model_mfd,
    get_obs_mfd,
    get_model_annual_eq_rate,
    get_total_obs_eqs,
)
from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
)
from openquake.hme.model_test_frameworks.relm.relm_stats import (
    bin_observance_log_likelihood,
)


def s_test_gdf_series(bin_gdf: GeoDataFrame, test_config: dict, N_norm: float = 1.0):
    return [
        s_test_bin(row.SpacemagBin, test_config, N_norm)
        for i, row in bin_gdf.iterrows()
    ]


def s_test_bin(sbin: SpacemagBin, test_cfg: dict, N_norm: float = 1.0):
    t_yrs = test_cfg["investigation_time"]
    like_fn = S_TEST_FN[test_cfg["likelihood_fn"]]

    # calculate the rate
    rate_mfd = sbin.get_rupture_mfd()
    rate_mfd = {mag: t_yrs * rate * N_norm for mag, rate in rate_mfd.items()}

    # calculate the observed L
    obs_eqs = sbin.observed_earthquakes
    obs_L = like_fn(rate_mfd, binned_events=obs_eqs)

    stoch_rup_counts = [
        get_poisson_counts_from_mfd(rate_mfd).copy() for i in range(test_cfg["n_iters"])
    ]

    # calculate L for iterated stochastic event sets
    stoch_Ls = np.array(
        [
            like_fn(rate_mfd, empirical_mfd=stoch_rup_counts[i],)
            for i in range(test_cfg["n_iters"])
        ]
    )

    return obs_L, stoch_Ls


def get_poisson_counts_from_mfd(mfd: dict):
    return {mag: np.random.poisson(rate) for mag, rate in mfd.items()}


def mfd_log_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
) -> float:
    """
    Calculates the log-likelihood of the observations (either `binned_events`
    or `empirical_mfd`) given the modeled rates (`rate_mfd`). The returned
    value is the log-likelihood of the whole MFD, which is the sum of the
    log-likelihoods of each bin, calculated using Poisson statistics.
    """
    if binned_events is not None:
        if empirical_mfd is None:
            num_obs_events = {mag: len(obs_eq) for mag, obs_eq in binned_events.items()}
        else:
            raise ValueError("Either use empirical_mfd or binned_events")
    else:
        num_obs_events = {mag: int(rate) for mag, rate in empirical_mfd.items()}

    return np.sum(
        [
            bin_observance_log_likelihood(n_obs, rate_mfd[mag])
            for mag, n_obs in num_obs_events.items()
        ]
    )


def total_event_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
) -> float:
    """
    Calculates the log-likelihood of the observations (either `binned_events`
    or `empirical_mfd`) given the modeled rates (`rate_mfd`). The returned
    value is the log-likelihood of the whole MFD, which is the sum of the
    log-likelihoods of each bin, calculated using Poisson statistics.
    """
    if binned_events is not None:
        if empirical_mfd is None:
            num_obs_events = {mag: len(obs_eq) for mag, obs_eq in binned_events.items()}
        else:
            raise ValueError("Either use empirical_mfd or binned_events")
    else:
        num_obs_events = {mag: int(rate) for mag, rate in empirical_mfd.items()}

    total_model_rate = sum(rate_mfd.values())
    total_num_events = sum(num_obs_events.values())

    # breakpoint()

    return bin_observance_log_likelihood(total_num_events, total_model_rate)


S_TEST_FN = {"n_eqs": total_event_likelihood, "mfd": mfd_log_likelihood}


def subdivide_observed_eqs(bin_gdf: GeoDataFrame, subcat_n_years: int):

    # collate earthquakes from bins
    obs_eqs = get_total_obs_eqs(bin_gdf, prospective=False)

    # divide earthquakes into groups, starting with the first observed year.
    # this could be changed to account for years with no events bounding the
    # catalog, but that would mean refactoring of the input yaml.
    obs_eqs.sort(key=lambda x: x.time)

    first_year = obs_eqs[0].time.year
    interval_start = first_year * 1
    last_year = obs_eqs[-1].time.year

    n_eqs = []
    while (interval_start + subcat_n_years) <= last_year:
        interval_end = interval_start + subcat_n_years
        n_eqs.append(
            len(
                [
                    eq
                    for eq in obs_eqs
                    if (interval_start <= eq.time.year <= interval_end)
                ]
            )
        )
        interval_start += subcat_n_years + 1

    return n_eqs


def N_test_poisson(
    num_obs_events: int, rupture_rate: float, conf_interval: float
) -> dict:

    conf_min, conf_max = poisson(rupture_rate).interval(conf_interval)

    test_pass = conf_min <= num_obs_events <= conf_max

    test_res = "Pass" if test_pass else "Fail"
    logging.info(f"N-Test: {test_res}")

    test_result = {
        "conf_interval_pct": conf_interval,
        "conf_interval": (conf_min, conf_max),
        "inv_time_rate": rupture_rate,
        "n_obs_earthquakes": num_obs_events,
        "pass": test_pass,
    }

    return test_result


def N_test_neg_binom(
    num_obs_events: int,
    rupture_rate: float,
    prob_success: float,
    r_dispersion: float,
    conf_interval: float,
) -> dict:

    if r_dispersion < 1:
        logging.warn(
            "Earthquake production temporally underdispersed, \n"
            "switching to Poisson N-Test"
        )
        return N_test_poisson(num_obs_events, rupture_rate, conf_interval)

    conf_min, conf_max = nbinom(r_dispersion, prob_success).interval(conf_interval)
    test_pass = conf_min <= num_obs_events <= conf_max

    test_res = "Pass" if test_pass else "Fail"
    logging.info(f"N-Test: {test_res}")

    test_result = {
        "conf_interval_pct": conf_interval,
        "conf_interval": (conf_min, conf_max),
        "inv_time_rate": rupture_rate,
        "n_obs_earthquakes": num_obs_events,
        "pass": test_pass,
    }

    return test_result
