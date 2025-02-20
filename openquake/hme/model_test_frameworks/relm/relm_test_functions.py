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
import pandas as pd
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
    breakpoint,
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
            "obs_loglike_total": obs_like_total,
            "stoch_loglike_totals": stoch_like_totals,
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
    normalize_n_eqs: Optional[bool] = True,
):
    # normalized to duration !!

    mod_mfd = get_model_mfd(
        rup_gdf, mag_bins, t_yrs=t_yrs, completeness_table=completeness_table
    )
    obs_mfd = get_obs_mfd(
        eq_gdf,
        mag_bins,
        # t_yrs=t_yrs,
        # completeness_table=completeness_table,
        stop_date=stop_date,
        annualize=False,
    )

    if normalize_n_eqs:
        N_obs = sum(obs_mfd.values())
        N_pred = sum(mod_mfd.values())
        N_norm = N_obs / N_pred
    else:
        N_norm = 1.0

    mod_mfd_norm = {k: v * N_norm for k, v in mod_mfd.items()}

    # calculate log-likelihoods
    n_bins = len(mod_mfd.keys())

    # should this be from mod_mfd_norm??
    stochastic_eq_counts = {
        bc: np.random.poisson(rate, size=n_iters)
        for bc, rate in mod_mfd_norm.items()
    }

    bin_log_likelihoods = {
        bc: [
            poisson_log_likelihood(
                n_stoch,
                (mod_mfd_norm[bc]),
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
                poisson_log_likelihood(obs_mfd[bc], rate)
                for bc, rate in mod_mfd.items()
            ]
        )
        / n_bins
    )

    pctile = (
        len(stoch_geom_mean_likes[stoch_geom_mean_likes <= obs_geom_mean_like])
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
            "stochastic_eq_counts": {
                k: v.tolist() for k, v in stochastic_eq_counts.items()
            },
            "model_mfd": mod_mfd,
            "model_mfd_norm": mod_mfd_norm,
            "obs_mfd": {k: float(v) for k, v in obs_mfd.items()},
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
    normalize_n_eqs: Optional[bool] = True,
    completeness_table: Optional[Sequence[Sequence[float]]] = None,
    stop_date: Optional[datetime] = None,
    critical_pct: float = 0.25,
    not_modeled_likelihood: float = 0.0,
    parallel: bool = False,
):
    # annual_rup_rate = rup_gdf.occurrence_rate.sum()
    if normalize_n_eqs:
        obs_mfd = get_obs_mfd(
            eq_gdf,
            mag_bins,
            t_yrs=t_yrs,
            stop_date=stop_date,
            completeness_table=completeness_table,
            annualize=False,
            cumulative=False,
        )
        N_obs = sum(obs_mfd.values())

        model_mfd = get_model_mfd(
            rup_gdf,
            mag_bins,
            t_yrs=t_yrs,
            completeness_table=completeness_table,
            cumulative=False,
        )
        N_pred = sum(model_mfd.values())
        N_norm = N_obs / N_pred
    else:
        N_norm = 1.0

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

    unmatched_eq_list = [
        cell_likes[cell]["unmatched_eqs"]
        for cell in cells
        if len(cell_likes[cell]["unmatched_eqs"]) > 0
    ]

    if len(unmatched_eq_list) > 0:
        unmatched_eqs = pd.concat(
            unmatched_eq_list,
            axis=0,
        )

        del unmatched_eqs["geometry"]
        unmatched_eqs = pd.DataFrame(unmatched_eqs)

    else:
        unmatched_eqs = []

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
        "unmatched_eqs": unmatched_eqs,
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
    """"""
    cell_id = rup_gdf.cell_id.values[0]
    t_yrs = test_cfg["investigation_time"]
    completeness_table = test_cfg.get("completeness_table")
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

    rate_mfd = get_model_mfd(
        rup_gdf, mag_bins, t_yrs=t_yrs, completeness_table=completeness_table
    )

    rate_mfd = {mag: rate for mag, rate in rate_mfd.items()}

    # eq catalog is already trimmed to investigation period
    obs_mfd = get_obs_mfd(
        eq_gdf,
        mag_bins,
        t_yrs=1.0,  # integrated over the investigation time, not annualized
        stop_date=stop_date,
        annualize=False,
    )
    likelihood_results = like_fn(
        rate_mfd,
        empirical_mfd=obs_mfd,
        N_norm=N_norm,
        return_likes=True,
        return_data=True,
        not_modeled_likelihood=not_modeled_likelihood,
    )

    obs_L = likelihood_results["bin_obs_log_like"]
    likes = likelihood_results["stoch_likes"]

    # handle bins with eqs but no rups
    bad_bins = []
    unmatched_eqs = []
    for like in likes:
        if like == not_modeled_log_like:
            bad_bins.append(cell_id)
            bin_ctr = h3.h3_to_geo(cell_id)
            bin_ctr = (round(bin_ctr[0], 3), round(bin_ctr[1], 3))
            logging.warning(f"{cell_id} {bin_ctr} has zero likelihood")
            for mag, rate in rate_mfd.items():
                if rate == 0.0 and obs_mfd[mag] > 0.0:
                    logging.warning(
                        f"mag bin {mag} has obs eqs but no ruptures"
                    )
                unmatched_eqs.append(eq_gdf[eq_gdf.mag_bin == mag])

    if len(unmatched_eqs) > 0:
        unmatched_eqs = pd.concat(unmatched_eqs, axis=0)

    stoch_rup_counts = get_poisson_counts_from_mfd_iter(
        rate_mfd, test_cfg["n_iters"]
    )

    # calculate L for iterated stochastic event sets
    stoch_Ls = np.array(
        [
            like_fn(
                rate_mfd,
                empirical_mfd=stoch_rup_counts[i],
                not_modeled_likelihood=not_modeled_likelihood,
            )["bin_obs_log_like"]
            for i in range(test_cfg["n_iters"])
        ]
    )

    return {
        "obs_loglike": obs_L,
        "stoch_loglikes": stoch_Ls,
        "bad_bins": bad_bins,
        "unmatched_eqs": unmatched_eqs,
        "obs_rate": likelihood_results["obs_rate"],
        "mod_rate": likelihood_results["mod_rate"],
        "obs_mfd": likelihood_results["obs_mfd"],
        "mod_mfd": likelihood_results["mod_mfd"],
    }


def n_test_function(rup_gdf, eq_gdf, test_config: dict):
    prospective = test_config.get("prospective", False)

    conf_interval = test_config.get("conf_interval", 0.95)

    if test_config.get("completeness_table"):
        model_mfd = get_model_mfd(
            rup_gdf,
            test_config["mag_bins"],
            completeness_table=test_config["completeness_table"],
        )
        test_rup_rate = sum(model_mfd.values())

    else:
        model_mfd = get_model_mfd(rup_gdf, test_config["mag_bins"], t_yrs=1.0)
        annual_rup_rate = rup_gdf.occurrence_rate.sum()
        test_rup_rate = annual_rup_rate * test_config["investigation_time"]

    M_min = min(model_mfd.keys())

    n_obs = len(eq_gdf)

    if test_config["prob_model"] == "poisson":
        test_result = N_test_poisson(n_obs, test_rup_rate, conf_interval)

    elif test_config["prob_model"] == "poisson_cum":
        n_iters = test_config.get("n_iters", 1000)  # will add to cfg
        n_eq_samples = [
            sum(get_poisson_counts_from_mfd(model_mfd).values())
            for _ in range(n_iters)
        ]
        test_result = N_test_empirical(n_obs, n_eq_samples, conf_interval)

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

    # for plotting
    test_result["M_min"] = M_min
    test_result["prob_model"] = test_config["prob_model"]

    return test_result


def get_poisson_counts_from_mfd(mfd: dict):
    return {mag: np.random.poisson(rate) for mag, rate in mfd.items()}


def mfd_log_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
    N_norm: float = 1.0,
    not_modeled_likelihood: float = 0.0,
    return_likes: bool = False,
    return_data: bool = False,
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

    likes = [
        bin_observance_log_likelihood(
            n_obs, rate_mfd[mag] * N_norm, not_modeled_likelihood
        )
        for mag, n_obs in num_obs_events.items()
    ]

    outputs = {"bin_obs_log_like": np.sum(likes)}

    if return_likes:
        outputs["stoch_likes"] = likes
    if return_data:
        outputs["obs_mfd"] = num_obs_events
        outputs["mod_mfd"] = rate_mfd
        outputs["mod_rate"] = total_model_rate
        outputs["obs_rate"] = total_num_events

    return outputs


def total_event_likelihood(
    rate_mfd: dict,
    binned_events: Optional[dict] = None,
    empirical_mfd: Optional[dict] = None,
    N_norm: float = 1.0,
    not_modeled_likelihood: float = 0.0,
    return_likes: bool = False,
    return_data: bool = False,
) -> float:
    """
    Calculates the log-likelihood of the observations (either `binned_events`
    or `empirical_mfd`) given the modeled rates (`rate_mfd`). The returned
    value is the log-likelihood of the total number of events compared to
    the modeled number of events, calculated using Poisson statistics.
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

    bin_obs_log_like = bin_observance_log_likelihood(
        total_num_events,
        total_model_rate * N_norm,
        not_modeled_val=not_modeled_likelihood,
    )

    outputs = {"bin_obs_log_like": bin_obs_log_like}
    if return_likes:
        outputs["stoch_likes"] = [bin_obs_log_like]
    if return_data:
        outputs["obs_mfd"] = num_obs_events
        outputs["mod_mfd"] = rate_mfd
        outputs["mod_rate"] = total_model_rate
        outputs["obs_rate"] = total_num_events

    return outputs


def N_test_empirical(
    num_obs_events: int, num_pred_events: Sequence[int], conf_interval: float
) -> dict:
    rupture_rate = np.mean(num_pred_events)
    conf_half_interval = conf_interval / 2
    conf_min = np.percentile(num_pred_events, 100 * (0.5 - conf_half_interval))
    conf_max = np.percentile(num_pred_events, 100 * (0.5 + conf_half_interval))

    test_pass = conf_min <= num_obs_events <= conf_max
    test_res = "Pass" if test_pass else "Fail"
    logging.info(f"N-Test: {test_res}")

    test_result = {
        "conf_interval_pct": conf_interval,
        "conf_interval": (conf_min, conf_max),
        "pred_samples": num_pred_events,
        "n_pred_earthquakes": rupture_rate,
        "n_obs_earthquakes": num_obs_events,
        "test_res": test_res,
        "test_pass": bool(test_pass),
    }

    return test_result


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
        "n_pred_earthquakes": rupture_rate,
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
