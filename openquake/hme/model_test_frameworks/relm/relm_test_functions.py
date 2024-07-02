"""
Utility functions for running tests in the RELM model test framework.
"""

import logging
from itertools import chain
from multiprocessing import Pool
from typing import Sequence, Optional
from datetime import datetime, timedelta

import h3
import numpy as np
from pandas import DataFrame
from geopandas import GeoDataFrame
from scipy.stats import poisson, nbinom
from numpy.lib.arraysetops import unique

# from openquake.hme.utils.bins import SpacemagBin
from openquake.hme.utils.utils import (
    get_model_mfd,
    get_obs_mfd,
    # get_model_annual_eq_rate,
    # get_total_obs_eqs,
    # get_n_eqs_from_mfd,
    get_poisson_counts_from_mfd_iter,
    _n_procs,
    get_cell_eqs,
)
from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
    poisson_log_likelihood,
)
from openquake.hme.model_test_frameworks.relm.relm_stats import (
    bin_observance_log_likelihood,
)


def l_test_function(
    # bin_gdf: GeoDataFrame,
    rup_gdf: GeoDataFrame,
    eq_gdf: GeoDataFrame,
    cell_groups,
    eq_groups,
    t_yrs: float,
    n_iters: int,
    mag_bins,
    completeness_table: Optional[Sequence[Sequence[float]]] = None,
    stop_date: Optional[datetime] = None,
    critical_pct: float = 0.25,
    not_modeled_likelihood: float = 0.0,
    append_results: bool = False,
):
    cell_like_cfg = {
        "investigation_time": t_yrs,
        "completeness_table": completeness_table,
        "stop_date": stop_date,
        "likelihood_fn": "mfd",
        "not_modeled_likelihood": not_modeled_likelihood,
        "n_iters": n_iters,
        "N_norm": 1.0,
        "mag_bins": mag_bins,
    }

    cell_likes = s_test_cells(
        cell_groups, rup_gdf, eq_groups, eq_gdf, cell_like_cfg
    )

    cells = sorted(cell_likes.keys())

    obs_likes = np.array([cell_likes[cell]["obs_loglike"] for cell in cells])
    stoch_likes = np.vstack(
        [cell_likes[cell]["stoch_loglikes"] for cell in cells]
    ).T
    bad_bins = list(
        unique(list(chain(*[cell_likes[cell]["bad_bins"] for cell in cells])))
    )

    obs_like_total = sum(obs_likes)
    stoch_like_totals = np.sum(stoch_likes, axis=1)

    pctile = (
        len(stoch_like_totals[stoch_like_totals <= obs_like_total]) / n_iters
    )

    test_pass = True if pctile >= critical_pct else False
    test_res = "Pass" if test_pass else "Fail"

    test_result = {
        "critical_pct": critical_pct,
        "percentile": pctile,
        "test_pass": bool(test_pass),
        "test_res": test_res,
        "bad_bins": bad_bins,
        "test_data": {
            "obs_loglike": obs_likes,
            "stoch_loglike": stoch_likes,
        },
    }

    return test_result


def m_test_function(
    rup_gdf,
    eq_gdf,
    mag_bins: dict,
    t_yrs: float,
    n_iters: int,
    completeness_table: Optional[Sequence[Sequence[float]]] = None,
    stop_date: Optional[datetime] = None,
    not_modeled_likelihood: float = 0.0,
    critical_pct: float = 0.25,
):
    mod_mfd = get_model_mfd(rup_gdf, mag_bins)
    obs_mfd = get_obs_mfd(
        eq_gdf,
        mag_bins,
        t_yrs=t_yrs,
        completeness_table=completeness_table,
        stop_date=stop_date,
    )

    # calculate log-likelihoods
    n_bins = len(mod_mfd.keys())

    stochastic_eq_counts = {
        bc: np.random.poisson((rate * t_yrs), size=n_iters)
        for bc, rate in mod_mfd.items()
    }

    bin_log_likelihoods = {
        bc: [
            poisson_log_likelihood(
                n_stoch,
                (mod_mfd[bc] * t_yrs),
                not_modeled_val=not_modeled_likelihood,
            )
            for n_stoch in eq_counts
        ]
        for bc, eq_counts in stochastic_eq_counts.items()
    }

    stoch_geom_mean_likes = np.array(
        [
            np.exp(
                np.sum(
                    [bll_mag[i] for bll_mag in bin_log_likelihoods.values()]
                )
                / n_bins
            )
            for i in range(n_iters)
        ]
    )

    obs_geom_mean_like = np.exp(
        np.sum(
            [
                poisson_log_likelihood(
                    int(obs_mfd[bc] * t_yrs),
                    rate * t_yrs,
                )
                for bc, rate in mod_mfd.items()
            ]
        )
        / n_bins
    )

    pctile = (
        len(stoch_geom_mean_likes[stoch_geom_mean_likes >= obs_geom_mean_like])
        / n_iters
    )

    test_pass = True if pctile >= critical_pct else False
    test_res = "Pass" if test_pass else "Fail"

    test_result = {
        "critical_pct": critical_pct,
        "percentile": pctile,
        "test_pass": test_pass,
        "test_res": test_res,
        "test_data": {
            "stoch_geom_mean_likes": stoch_geom_mean_likes.tolist(),
            "obs_geom_mean_like": obs_geom_mean_like,
        },
    }

    return test_result


def s_test_function(
    rup_gdf: GeoDataFrame,
    eq_gdf: GeoDataFrame,
    cell_groups,
    eq_groups,
    t_yrs: float,
    n_iters: int,
    likelihood_fn: str,
    mag_bins,
    completeness_table: Optional[Sequence[Sequence[float]]] = None,
    stop_date: Optional[datetime] = None,
    critical_pct: float = 0.25,
    not_modeled_likelihood: float = 0.0,
    parallel: bool = False,
):
    annual_rup_rate = rup_gdf.occurrence_rate.sum()

    N_obs = len(eq_gdf)
    N_pred = annual_rup_rate * t_yrs
    N_norm = N_obs / N_pred

    cell_like_cfg = {
        "investigation_time": t_yrs,
        "likelihood_fn": likelihood_fn,
        "not_modeled_likelihood": not_modeled_likelihood,
        "n_iters": n_iters,
        "N_norm": N_norm,
        "mag_bins": mag_bins,
        "completeness_table": completeness_table,
        "stop_date": stop_date,
    }

    cell_likes = s_test_cells(
        cell_groups,
        rup_gdf,
        eq_groups,
        eq_gdf,
        cell_like_cfg,
        parallel=parallel,
    )

    cells = sorted(cell_likes.keys())

    obs_likes = np.array([cell_likes[cell]["obs_loglike"] for cell in cells])
    stoch_likes = np.vstack(
        [cell_likes[cell]["stoch_loglikes"] for cell in cells]
    )
    bad_bins = list(
        unique(list(chain(*[cell_likes[cell]["bad_bins"] for cell in cells])))
    )

    cell_fracs = np.zeros(len(cells))

    for i, obs_like in enumerate(obs_likes):
        cell_stoch_likes = stoch_likes[i]
        cell_fracs[i] = sum(cell_stoch_likes <= obs_like) / n_iters

    obs_like_total = sum(obs_likes)
    stoch_like_totals = np.sum(stoch_likes, axis=0)

    pctile = sum(stoch_like_totals <= obs_like_total) / n_iters

    test_pass = True if pctile >= critical_pct else False
    test_res = "Pass" if test_pass else "Fail"

    test_result = {
        "critical_pct": critical_pct,
        "percentile": pctile,
        "test_pass": bool(test_pass),
        "test_res": test_res,
        "bad_bins": bad_bins,
        "test_data": {
            "obs_loglike": obs_likes,
            "stoch_loglike": stoch_likes,
            "cell_loglikes": cell_likes,
            "cell_fracs": cell_fracs,
        },
    }

    return test_result


def s_test_cells(
    cell_groups, rup_gdf, eq_groups, eq_gdf, test_cfg, parallel: bool = False
):
    s_test_cell_results = {}

    cell_ids = sorted(cell_groups.groups.keys())

    args = (
        (
            rup_gdf.loc[cell_groups.groups[cell_id]],
            get_cell_eqs(cell_id, eq_gdf, eq_groups),
            test_cfg,
        )
        for cell_id in cell_ids
    )

    # need to make parallel processing optional
    if parallel is True:
        with Pool(_n_procs) as p:
            s_test_cell_results_ = p.map(_s_test_cell_args, args)
    else:
        s_test_cell_results_ = list(map(_s_test_cell_args, args))

    s_test_cell_results = {
        cell_id: s_test_cell_results_[i] for i, cell_id in enumerate(cell_ids)
    }

    return s_test_cell_results


def _s_test_cell_args(cell_args):
    return s_test_cell(*cell_args)


def s_test_cell(rup_gdf, eq_gdf, test_cfg):
    cell_id = rup_gdf.cell_id.values[0]

    t_yrs = test_cfg["investigation_time"]
    completeness_table = test_cfg["completeness_table"]
    mag_bins = test_cfg["mag_bins"]
    stop_date = test_cfg["stop_date"]
    like_fn = S_TEST_FN[test_cfg["likelihood_fn"]]
    not_modeled_likelihood = test_cfg["not_modeled_likelihood"]
    N_norm = test_cfg["N_norm"]
    not_modeled_log_like = (
        -np.inf
        if not_modeled_likelihood == 0.0
        else np.log(not_modeled_likelihood)
    )

    rate_mfd = get_model_mfd(rup_gdf, mag_bins)
    rate_mfd = {mag: t_yrs * rate * N_norm for mag, rate in rate_mfd.items()}

    obs_mfd = get_obs_mfd(
        eq_gdf,
        mag_bins,
        t_yrs=t_yrs,
        completeness_table=completeness_table,
        stop_date=stop_date,
    )
    obs_L, likes = like_fn(rate_mfd, empirical_mfd=obs_mfd, return_likes=True)

    # handle bins with eqs but no rups
    bad_bins = []
    for like in likes:
        if like == not_modeled_log_like:
            bad_bins.append(cell_id)
            bin_ctr = h3.h3_to_geo(cell_id)
            bin_ctr = (round(bin_ctr[0], 3), round(bin_ctr[1], 3))
            logging.warn(f"{cell_id} {bin_ctr} has zero likelihood")
            for mag, rate in rate_mfd.items():
                if rate == 0.0 and obs_mfd[mag] > 0.0:
                    logging.warn(f"mag bin {mag} has obs eqs but no ruptures")

    stoch_rup_counts = get_poisson_counts_from_mfd_iter(
        rate_mfd, test_cfg["n_iters"]
    )

    # should come up with vectorized likelihood functions (might work already)
    # with proper setup

    # calculate L for iterated stochastic event sets
    stoch_Ls = np.array(
        [
            like_fn(
                rate_mfd,
                empirical_mfd=stoch_rup_counts[i],
                not_modeled_likelihood=not_modeled_likelihood,
            )
            for i in range(test_cfg["n_iters"])
        ]
    )

    return {
        "obs_loglike": obs_L,
        "stoch_loglikes": stoch_Ls,
        "bad_bins": bad_bins,
    }


def n_test_function(rup_gdf, eq_gdf, test_config: dict):
    prospective = test_config.get("prospective", False)

    conf_interval = test_config.get("conf_interval", 0.95)

    annual_rup_rate = rup_gdf.occurrence_rate.sum()
    n_obs = len(eq_gdf)

    test_rup_rate = annual_rup_rate * test_config["investigation_time"]

    if test_config["prob_model"] == "poisson":
        test_result = N_test_poisson(n_obs, test_rup_rate, conf_interval)

    elif test_config["prob_model"] == "neg_binom":
        raise NotImplementedError("can't subdivide earthquakes yet")
        n_eqs_in_subs = subdivide_observed_eqs(
            bin_gdf, test_config["investigation_time"]
        )

        if prospective:
            prob_success, r_dispersion = estimate_negative_binom_parameters(
                n_eqs_in_subs, test_rup_rate
            )
        else:
            prob_success, r_dispersion = estimate_negative_binom_parameters(
                n_eqs_in_subs
            )

        test_result = N_test_neg_binom(
            n_obs,
            test_rup_rate,
            prob_success,
            r_dispersion,
            test_config["conf_interval"],
        )

    return test_result


def get_poisson_counts_from_mfd(mfd: dict):
    return {mag: np.random.poisson(rate) for mag, rate in mfd.items()}


def mfd_log_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
    not_modeled_likelihood: float = 0.0,
    return_likes: bool = False,
) -> float:
    """
    Calculates the log-likelihood of the observations (either `binned_events`
    or `empirical_mfd`) given the modeled rates (`rate_mfd`). The returned
    value is the log-likelihood of the whole MFD, which is the sum of the
    log-likelihoods of each bin, calculated using Poisson statistics.
    """
    if binned_events is not None:
        if empirical_mfd is None:
            num_obs_events = {
                mag: len(obs_eq) for mag, obs_eq in binned_events.items()
            }
        else:
            raise ValueError("Either use empirical_mfd or binned_events")
    else:
        num_obs_events = {
            mag: int(rate) for mag, rate in empirical_mfd.items()
        }

    likes = [
        bin_observance_log_likelihood(
            n_obs, rate_mfd[mag], not_modeled_likelihood
        )
        for mag, n_obs in num_obs_events.items()
    ]

    if return_likes:
        return np.sum(likes), likes
    else:
        return np.sum(likes)


def total_event_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
    not_modeled_likelihood: float = 0.0,
) -> float:
    """
    Calculates the log-likelihood of the observations (either `binned_events`
    or `empirical_mfd`) given the modeled rates (`rate_mfd`). The returned
    value is the log-likelihood of the whole MFD, which is the sum of the
    log-likelihoods of each bin, calculated using Poisson statistics.
    """
    if binned_events is not None:
        if empirical_mfd is None:
            num_obs_events = {
                mag: len(obs_eq) for mag, obs_eq in binned_events.items()
            }
        else:
            raise ValueError("Either use empirical_mfd or binned_events")
    else:
        num_obs_events = {
            mag: int(rate) for mag, rate in empirical_mfd.items()
        }

    total_model_rate = sum(rate_mfd.values())
    total_num_events = sum(num_obs_events.values())

    return bin_observance_log_likelihood(
        total_num_events,
        total_model_rate,
        not_modeled_val=not_modeled_likelihood,
    )


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
        "test_res": test_res,
        "test_pass": bool(test_pass),
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

    conf_min, conf_max = nbinom(r_dispersion, prob_success).interval(
        conf_interval
    )
    test_pass = conf_min <= num_obs_events <= conf_max

    test_res = "Pass" if test_pass else "Fail"
    logging.info(f"N-Test: {test_res}")

    test_result = {
        "conf_interval_pct": conf_interval,
        "conf_interval": (conf_min, conf_max),
        "inv_time_rate": rupture_rate,
        "n_obs_earthquakes": num_obs_events,
        "test_res": test_res,
        "test_pass": bool(test_pass),
    }

    return test_result


S_TEST_FN = {"n_eqs": total_event_likelihood, "mfd": mfd_log_likelihood}

"""
def subdivide_observed_eqs_old(bin_gdf: GeoDataFrame, subcat_n_years: int):

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


def subdivide_observed_eqs(eq_gdf, subcat_n_years, t_yrs, start_year=None):

    if start_year is None:
        pass

"""
