"""
Utility functions for running tests in the RELM model test framework.
"""
import logging
from typing import Sequence, Dict, List
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import poisson, nbinom
from geopandas import GeoSeries, GeoDataFrame

from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
)


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
