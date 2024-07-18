import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils import get_mag_bins_from_cfg, deep_update
from ..sanity.sanity_checks import max_check
from .gem_test_functions import (
    # get_stochastic_mfd,
    # get_stochastic_mfds_parallel,
    # eval_obs_moment,
    # eval_obs_moment_model,
    model_mfd_eval_fn,
    moment_over_under_eval_fn,
    rupture_matching_eval_fn,
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
    (specified by the `critical_pct` argument), then the test fails.

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
    critical_pct = test_config.get("critical_pct", 0.25)
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]
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
        critical_pct=critical_pct,
    )

    logging.info("M-Test crit pct {}".format(test_result["critical_pct"]))
    logging.info("M-Test pct {}".format(test_result["percentile"]))
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
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    test_config["parallel"] = cfg["config"]["parallel"]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]
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
        completeness_table=completeness_table,
        stop_date=stop_date,
        mag_bins=mag_bins,
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("S-Test {}".format(test_results["test_res"]))
    logging.info("S-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("S-Test model pct: {}".format(test_results["percentile"]))
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
        t_yrs = cfg["input"]["seis_catalog"]["duration"]
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
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("L-Test {}".format(test_results["test_res"]))
    logging.info("L-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("L-Test model pct: {}".format(test_results["percentile"]))
    return test_results


def N_test(cfg: dict, input_data: dict) -> dict:
    logging.info("Running N-Test")

    test_config = cfg["config"]["model_framework"]["gem"]["N_test"]

    prospective = test_config.get("prospective", False)

    if (test_config["prob_model"] == "poisson") and not prospective:
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
        "N-Test number pred eqs: {}".format(test_results["inv_time_rate"])
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
    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["model_mfd"]
    prospective = test_config.get("prospective", False)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    results = model_mfd_eval_fn(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        test_config["investigation_time"],
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


def rupture_matching_eval(cfg, input_data):
    logging.info("Running GEM Rupture Matching Eval")

    default_params = {
        "distance_lambda": 10.0,
        "mag_window": 1.0,
        "group_return_threshold": 0.9,
        "min_likelihood": 0.1,
        "no_attitude_default_like": 0.5,
        "no_rake_default_like": 0.5,
        "use_occurrence_rate": False,
        "return_one": "best",
        "parallel": False,
    }

    test_config = cfg["config"]["model_framework"]["gem"][
        "rupture_matching_eval"
    ]
    prospective = test_config.get("prospective", False)

    test_config = deep_update(default_params, test_config)

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
        parallel=cfg["config"]["parallel"],
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
    logging.warn("GEM Likelihood test deprecated")
    return


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
}
