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
from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
)
from openquake.hme.model_test_frameworks.relm.relm_stats import (
    bin_observance_log_likelihood,
)


def s_test_gdf_series(bin_gdf: GeoDataFrame, test_config: dict, 
                      N_norm: float = 1.0):
    return [
        s_test_bin(row.SpacemagBin, test_config, N_norm)
        for i, row in bin_gdf.iterrows()
    ]


def s_test_gdf():
    pass


def s_test_bin(sbin: SpacemagBin, test_cfg: dict, N_norm: float = 1.0):
    t_yrs = test_cfg["investigation_time"]

    # calculate the rate
    rate_mfd = sbin.get_rupture_mfd()
    rate_mfd = {mag: t_yrs * rate * N_norm for mag, rate in rate_mfd.items()}

    # calculate the observed L
    obs_eqs = sbin.observed_earthquakes
    #obs_L = mfd_log_likelihood(rate_mfd, obs_eqs)
    obs_L = total_event_likelihood(rate_mfd, binned_events=obs_eqs)

    stoch_rup_counts = [
        get_poisson_counts_from_mfd(rate_mfd)
            .copy() for i in range(test_cfg["n_iters"])
    ]

    # calculate L for iterated stochastic event sets
    stoch_Ls = np.array(
        [
            # mfd_log_likelihood(
            total_event_likelihood(
                rate_mfd,
                # binned_events=binned_event_arr[i],
                empirical_mfd=stoch_rup_counts[i],
            )
            for i in range(test_cfg["n_iters"])
        ]
    )

    # breakpoint()

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
            num_obs_events = {mag: len(obs_eq) 
                              for mag, obs_eq in binned_events.items()}
        else:
            raise ValueError("Either use empirical_mfd or binned_events")
    else:
        num_obs_events = {mag: int(rate) for mag, rate in empirical_mfd.items()}

    total_model_rate = sum(rate_mfd.values())
    total_num_events = sum(num_obs_events.values())

    # breakpoint()

    return bin_observance_log_likelihood(total_num_events, total_model_rate)


def get_model_mfd(bin_gdf: GeoDataFrame, cumulative: bool = False) -> dict:
    mod_mfd = bin_gdf.iloc[0].SpacemagBin.get_rupture_mfd()
    mag_bin_centers = bin_gdf.iloc[0].SpacemagBin.mag_bin_centers

    for i, row in bin_gdf.iloc[1:].iterrows():
        bin_mod_mfd = row.SpacemagBin.get_rupture_mfd()
        for bin_center, rate in bin_mod_mfd.items():
            mod_mfd[bin_center] += rate

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in mag_bin_centers[::-1]:
            cum_mag += mod_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        mod_mfd = {cb: cum_mfd[cb] for cb in mag_bin_centers}

    return mod_mfd


def get_obs_mfd(
    bin_gdf: GeoDataFrame,
    t_yrs: float,
    prospective: bool = False,
    cumulative: bool = False,
) -> dict:
    mag_bin_centers = bin_gdf.iloc[0].SpacemagBin.mag_bin_centers

    if prospective is False:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_empirical_mfd(t_yrs=t_yrs)
    else:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_prospective_mfd(t_yrs=t_yrs)

    for i, row in bin_gdf.iloc[1:].iterrows():
        if prospective is False:
            bin_obs_mfd = row.SpacemagBin.get_empirical_mfd(t_yrs=t_yrs)
        else:
            bin_obs_mfd = row.SpacemagBin.get_prospective_mfd(t_yrs=t_yrs)

        for bin_center, rate in bin_obs_mfd.items():
            obs_mfd[bin_center] += rate

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in mag_bin_centers[::-1]:
            cum_mag += obs_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        obs_mfd = {cb: cum_mfd[cb] for cb in mag_bin_centers}

    return obs_mfd


def get_model_annual_eq_rate(bin_gdf: GeoDataFrame) -> float:
    annual_rup_rate = 0.0
    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin
        min_bin_center = np.min(sb.mag_bin_centers)
        bin_mfd = sb.get_rupture_mfd(cumulative=True)
        annual_rup_rate += bin_mfd[min_bin_center]

    return annual_rup_rate


def get_total_obs_eqs(bin_gdf: GeoDataFrame, prospective: bool = False) -> list:
    """
    Returns a list of all of the observed earthquakes within the model domain.
    """
    obs_eqs = []

    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin

        if prospective is False:
            for mb in sb.observed_earthquakes.values():
                obs_eqs.extend(mb)
        else:
            for mb in sb.prospective_earthquakes.values():
                obs_eqs.extend(mb)

    return obs_eqs


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
