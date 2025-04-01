import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils import (
    get_mag_bins_from_cfg,
    deep_update,
    get_mag_year_from_comp_table,
    get_model_mfd,
)

from ..sanity.sanity_checks import max_check
from .gem_test_functions import (
    # get_stochastic_mfd,
    # get_stochastic_mfds_parallel,
    # eval_obs_moment,
    # eval_obs_moment_model,
    model_mfd_eval_fn,
    moment_over_under_eval_fn,
    rupture_matching_eval_fn,
    catalog_ground_motion_eval_fn,
)

from ..relm.relm_tests import (
    n_test_function,
    s_test_function,
    m_test_function,
    l_test_function,
)

from .gem_stats import calc_mfd_log_likelihood_independent


def M_test(
    cfg,
    input_data,
) -> dict:
    """
    The M-Test is based on Zechar et al. (2010), though not identical. This
    tests evaluates the consistency of the magnitude-frequency distribution of
    the model vs. the observations, by evaluating the log-likelihood of the
    observed earthquakes given the model (forecast), compared with the
    log-likelihood of a large number of stochastic catalogs generated from the
    same forecast. If the log-likelihood of the observed earthquake catalog is
    less than the majority of the log-likelihoods of stochastic catalogs
    (specified by the `critical_frac` argument), then the test fails.

    The log-likelihoods are calculated first for each magnitude bin. The
    log-likelihood for each magnitude bin is the log-likelihood of the observed
    (or stochastic) number of earthquakes in that magnitude bin occurring
    throughout the model domain, given the mean rupture rate for that magnitude
    bin, using the Poisson distribution.

    Then, the log-likelihoods of the observed catalog and the stochastic
    catalogs are calculated as the geometric mean of the individual bin
    likelihoods.

    The differences between this implementation and that of Zechar et al. (2010)
    is that 1) in this version we do not fix the total number of earthquakes
    that occurs in each stochastic simulation (because that is somewhat
    complicated to implement within Hamlet) and 2) we use the geometric mean
    instead of the product of the magnitude bin likelihoods for the total
    likelihood, because this lets us disregard the discretization of the MFD
    when comparing between different models. Note that in terms of passing or
    failing, (1) does not matter much if the model passes the N-test, and (2)
    does not matter at all because the ranking of the observed and stochasitc
    catalogs will remain the same.
    """
    logging.info("Running GEM M-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["M_test"]

    prospective = test_config.get("prospective", False)
    critical_frac = test_config.get("critical_frac", 0.25)
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)
    normalize_n_eqs = test_config.get("normalize_n_eqs", True)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        t_yrs = cfg["input"]["seis_catalog"].get("duration", 1.0)
        stop_date = cfg["input"]["seis_catalog"].get("stop_date")
        completeness_table = cfg["input"]["seis_catalog"].get(
            "completeness_table"
        )

    test_result = m_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs,
        test_config["n_iters"],
        completeness_table=completeness_table,
        stop_date=stop_date,
        not_modeled_likelihood=not_modeled_likelihood,
        critical_frac=critical_frac,
        normalize_n_eqs=normalize_n_eqs,
    )

    logging.info("M-Test crit frac {}".format(test_result["critical_frac"]))
    logging.info("M-Test fractile {}".format(test_result["fractile"]))
    logging.info("M-Test {}".format(test_result["test_res"]))
    return test_result


def S_test(
    cfg: dict,
    input_data: dict,
) -> dict:
    """"""
    logging.info("Running GEM S-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["S_test"]
    prospective = test_config.get("prospective", False)
    likelihood_function = test_config.get("likelihood_function", "mfd")
    normalize_n_eqs = test_config.get("normalize_n_eqs", False)
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)
    test_config["parallel"] = cfg["config"]["parallel"]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"].get("duration")
        stop_date = cfg["input"]["seis_catalog"].get("stop_date")
        completeness_table = cfg["input"]["seis_catalog"].get(
            "completeness_table"
        )

    test_results = s_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        likelihood_function,
        mag_bins=mag_bins,
        normalize_n_eqs=normalize_n_eqs,
        completeness_table=completeness_table,
        stop_date=stop_date,
        critical_frac=test_config["critical_frac"],
        not_modeled_likelihood=not_modeled_likelihood,
        parallel=test_config["parallel"],
    )

    logging.info("S-Test {}".format(test_results["test_res"]))
    logging.info("S-Test crit frac: {}".format(test_results["critical_frac"]))
    logging.info("S-Test model fractile: {}".format(test_results["fractile"]))
    return test_results


def L_test(
    cfg: dict,
    input_data: dict,
) -> dict:
    """"""
    logging.info("Running GEM L-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["L_test"]
    prospective = test_config.get("prospective", False)
    not_modeled_likelihood = 0.0  # hardcoded for RELM
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"].get("duration")
        stop_date = cfg["input"]["seis_catalog"].get("stop_date")
        completeness_table = cfg["input"]["seis_catalog"].get(
            "completeness_table"
        )

    test_results = l_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        mag_bins,
        completeness_table=completeness_table,
        stop_date=stop_date,
        critical_frac=test_config["critical_frac"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("L-Test {}".format(test_results["test_res"]))
    logging.info("L-Test crit frac: {}".format(test_results["critical_frac"]))
    logging.info("L-Test model fractile: {}".format(test_results["fractile"]))
    return test_results


def N_test(cfg: dict, input_data: dict) -> dict:
    logging.info("Running N-Test")

    test_config = cfg["config"]["model_framework"]["gem"]["N_test"]
    completeness_table = cfg["input"]["seis_catalog"].get("completeness_table")
    test_config["mag_bins"] = get_mag_bins_from_cfg(cfg)

    prospective = test_config.get("prospective", False)

    if (
        test_config["prob_model"] in ["poisson", "poisson_cum"]
    ) and not prospective:
        if completeness_table is not None:
            test_config["completeness_table"] = completeness_table
            test_config["mag_bins"] = get_mag_bins_from_cfg(cfg)
        else:
            inv_time = test_config.get("investigation_time")
            seis_duration = cfg["input"]["seis_catalog"]["duration"]
            if inv_time is not None and inv_time != seis_duration:
                logging.warning(
                    "N-Test: Investigation time does not match seis catalog "
                    "duration. Using seis catalog duration."
                )

            test_config["investigation_time"] = cfg["input"]["seis_catalog"][
                "duration"
            ]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    test_results = n_test_function(
        input_data["rupture_gdf"], eq_gdf, test_config
    )

    logging.info(
        "N-Test number obs eqs: {}".format(test_results["n_obs_earthquakes"])
    )
    logging.info(
        "N-Test number pred eqs: {}".format(test_results["n_pred_earthquakes"])
    )
    logging.info("N-Test {}".format(test_results["test_pass"]))
    return test_results


def max_mag_check(cfg: dict, input_data: dict):
    logging.info("Checking Maximum Magnitudes")

    max_bin_check_results = max_check(cfg, input_data, framework="gem")

    bad_bins = [
        cell
        for cell, max_check_val in max_bin_check_results.items()
        if max_check_val is False
    ]

    # could add all results here for the map...
    if bad_bins == []:
        results = {"test_res": "Pass", "test_pass": True, "bad_bins": bad_bins}
    else:
        results = {
            "test_res": "Fail",
            "test_pass": False,
            "bad_bins": bad_bins,
        }

    logging.info("Max Mag Check res: {}".format(results["test_res"]))
    return results


def model_mfd_eval(cfg, input_data):
    logging.info("Running GEM Model MFD Eval")
    mag_bins = get_mag_bins_from_cfg(cfg)
    completeness_table = cfg["input"]["seis_catalog"].get("completeness_table")
    test_config = cfg["config"]["model_framework"]["gem"]["model_mfd"]

    if test_config is None:
        test_config = {}

    prospective = test_config.get("prospective", False)
    test_config["investigation_time"] = test_config.get(
        "investigation_time", cfg["input"]["seis_catalog"].get("duration")
    )

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    results = model_mfd_eval_fn(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs=test_config["investigation_time"],
        completeness_table=completeness_table,
    )

    return results


def moment_over_under_eval(cfg, input_data):
    logging.info("Running GEM Moment Over-Under Eval")

    test_config = cfg["config"]["model_framework"]["gem"]["moment_over_under"]
    mag_bins = get_mag_bins_from_cfg(cfg)
    min_bin_mag = mag_bins[sorted(mag_bins.keys())[0]][0]
    max_bin_mag = mag_bins[sorted(mag_bins.keys())[-1]][1]

    prospective = test_config.get("prospective", False)
    t_yrs = test_config["investigation_time"]
    n_iters = test_config["n_iters"]
    min_mag = test_config.get("min_mag", min_bin_mag)
    max_mag = test_config.get("max_mag", max_bin_mag)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    test_results = moment_over_under_eval_fn(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        t_yrs,
        min_mag,
        max_mag,
        n_iters,
    )

    results_for_print = {
        "total_obs_moment": test_results["test_data"]["total_obs_moment"],
        "modeled_obs_moment_mean": test_results["test_data"][
            "modeled_obs_moment"
        ]["mean"],
        "modeled_obs_moment_sd": test_results["test_data"][
            "modeled_obs_moment"
        ]["sd"],
        "fractile": test_results["test_data"]["frac"],
    }

    logging.info("Moment Over-Under Results: {}".format(results_for_print))

    return test_results


rup_match_default_params = {
    "distance_lambda": 1.0,
    "mag_window": 1.0,
    "group_return_threshold": 0.9,
    "min_likelihood": 0.1,
    "no_attitude_default_like": 0.5,
    "no_rake_default_like": 0.5,
    "use_occurrence_rate": False,
    "return_one": "best",
    "parallel": False,
}


def rupture_matching_eval(cfg, input_data):
    logging.info("Running GEM Rupture Matching Eval")

    test_config = cfg["config"]["model_framework"]["gem"][
        "rupture_matching_eval"
    ]
    prospective = test_config.get("prospective", False)

    test_config = deep_update(rup_match_default_params, test_config)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    match_results = rupture_matching_eval_fn(
        input_data["rupture_gdf"],
        eq_gdf,
        distance_lambda=test_config["distance_lambda"],
        mag_window=test_config["mag_window"],
        group_return_threshold=test_config["group_return_threshold"],
        no_attitude_default_like=test_config["no_attitude_default_like"],
        no_rake_default_like=test_config["no_rake_default_like"],
        use_occurrence_rate=test_config["use_occurrence_rate"],
        return_one=test_config["return_one"],
        # parallel is often slower
        parallel=test_config["parallel"],  # cfg["config"]["parallel"],
    )

    n_unmatched = len(match_results["unmatched_eqs"])
    n_total = len(eq_gdf)
    mean_likelihood = np.round(
        match_results["matched_rups"].likelihood.mean(), 3
    )
    test_results_for_print = {
        "N total": n_total,
        "N Unmatched": n_unmatched,
        "Mean match likelihood": mean_likelihood,
    }

    match_results.update(
        {
            "num_matched": n_total - n_unmatched,
            "num_eq": n_total,
            "mean_match_likelihood": mean_likelihood,
        }
    )

    logging.info(
        "Rupture Matching Eval Results: {}".format(test_results_for_print)
    )

    test_results = match_results

    return test_results


def mfd_likelihood_test(cfg, input_data):
    logging.warning("GEM Likelihood test deprecated")
    return


def cumulative_occurrence_eval(cfg, input_data):
    logging.info("Running GEM Cumultive Earthquake Occurrence Eval")

    eqs = input_data["eq_gdf"]
    rup_gdf = input_data["rupture_gdf"]

    start_date = cfg["input"]["seis_catalog"].get("start_date")
    stop_date = cfg["input"]["seis_catalog"].get("stop_date")
    comp_table = cfg["input"]["seis_catalog"].get("completeness_table")

    mag_bins = get_mag_bins_from_cfg(cfg)

    if not comp_table:
        start_dates = {k: start_date for k in mag_bins.keys()}
        t_yrs = stop_date - start_date
    else:
        start_dates = {
            k: f"{get_mag_year_from_comp_table(comp_table, k)[1]}-01-01"
            for k in mag_bins.keys()
        }
        t_yrs = None

    eqs_by_mag_time = {
        k: {
            "start_date": start_dates[k],
            "stop_date": stop_date,
            "eqs": eqs[eqs.mag_bin == k].sort_values("time"),
        }
        for k in mag_bins.keys()
    }

    model_mfd = get_model_mfd(
        rup_gdf,
        mag_bins,
        t_yrs=t_yrs,
        completeness_table=comp_table,
        stop_date=stop_date,
    )

    return {
        "eqs_by_mag_time": eqs_by_mag_time,
        "model_mfd": model_mfd,
    }


def catalog_ground_motion_eval(cfg, input_data):

    logging.info("Running GEM catalog ground motion evaluation")

    test_config = cfg["config"]["model_framework"]["gem"][
        "catalog_ground_motion_eval"
    ]

    match_rups = test_config.get("match_rups", False)
    test_config["gmf_method"] = test_config.get(
        "gmf_method", "ground_motion_fields"
    )

    test_config = deep_update(rup_match_default_params, test_config)

    gmm_comparisons = catalog_ground_motion_eval_fn(test_config, input_data)

    return {"gmm_comparisons": gmm_comparisons}


gem_test_dict = {
    "likelihood": mfd_likelihood_test,
    "max_mag_check": max_mag_check,
    "model_mfd": model_mfd_eval,
    "moment_over_under": moment_over_under_eval,
    "M_test": M_test,
    "S_test": S_test,
    "N_test": N_test,
    "L_test": L_test,
    "rupture_matching_eval": rupture_matching_eval,
    "cumulative_occurrence_eval": cumulative_occurrence_eval,
    "catalog_ground_motion_eval": catalog_ground_motion_eval,
}
