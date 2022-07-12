import logging
from typing import Optional
from webbrowser import get

import numpy as np
from numpy.lib.function_base import append
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
)
from openquake.hme.utils import (
    get_mag_bins_from_cfg,
    # get_source_bins,
    # get_n_eqs_from_mfd,
)

from openquake.hme.utils.plots import plot_mfd
from openquake.hme.utils.stats import poisson_likelihood, poisson_log_likelihood
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    N_test_poisson,
    N_test_neg_binom,
    s_test_function,
    subdivide_observed_eqs,
    get_model_annual_eq_rate,
    get_total_obs_eqs,
    get_model_mfd,
    get_obs_mfd,
    # s_test_bin,
    s_test_gdf_series,
    m_test_function,
    s_test_function,
    n_test_function,
    l_test_function,
)


def M_test(cfg, input_data):
    logging.info("Running CSEP/RELM M-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["relm"]["M_test"]
    prospective = test_config.get("prospective", False)
    critical_pct = test_config.get("critical_pct", 0.25)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_result = m_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs,
        test_config["n_iters"],
        not_modeled_likelihood=0.0,
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
    logging.info("Running CSEP/RELM S-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["relm"]["S_test"]
    prospective = test_config.get("prospective", False)
    likelihood_function = test_config.get("likelihood_function", "mfd")
    not_modeled_likelihood = 0.0  # hardcoded for RELM

    parallel = cfg["config"]["parallel"]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_results = s_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        likelihood_function,
        mag_bins=mag_bins,
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
        parallel=parallel,
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
    logging.info("Running CSEP/RELM L-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["relm"]["L_test"]
    prospective = test_config.get("prospective", False)
    append_results = test_config.get("append")
    not_modeled_likelihood = 0.0  # hardcoded for RELM

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_results = l_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        mag_bins,
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
        append_results=append_results,
    )

    logging.info("L-Test {}".format(test_results["test_res"]))
    logging.info("L-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("L-Test model pct: {}".format(test_results["percentile"]))
    return test_results


def N_test(cfg: dict, input_data: dict) -> dict:
    logging.info("Running N-Test")

    test_config = cfg["config"]["model_framework"]["relm"]["N_test"]

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


"""
OBSOLETE
"""


def M_test_old(
    cfg,
    bin_gdf: Optional[GeoDataFrame] = None,
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
    does not matter at all because the ranking of the observed and stochastic
    catalogs will remain the same.
    """
    logging.info("Running CSEP/RELM M-Test")

    test_config = cfg["config"]["model_framework"]["relm"]["M_test"]

    prospective = test_config.get("prospective", False)

    critical_pct = test_config.get("critical_pct", 0.25)

    t_yrs = test_config["investigation_time"]

    test_result = m_test_function(
        bin_gdf,
        t_yrs,
        test_config["n_iters"],
        prospective=prospective,
        not_modeled_likelihood=0.0,
        critical_pct=critical_pct,
    )

    logging.info("M-Test crit pct {}".format(test_result["critical_pct"]))
    logging.info("M-Test pct {}".format(test_result["percentile"]))
    logging.info("M-Test {}".format(test_result["test_res"]))
    return test_result


def S_test_old(
    cfg: dict,
    bin_gdf: GeoDataFrame,
) -> dict:
    """"""
    logging.info("Running S-Test")

    test_config = cfg["config"]["model_framework"]["relm"]["S_test"]
    t_yrs = test_config["investigation_time"]
    prospective = test_config.get("prospective", False)
    append_results = test_config.get("append")
    test_config["not_modeled_likelihood"] = 0.0  # hardcoded for RELM

    test_results = s_test_function(
        bin_gdf,
        t_yrs,
        test_config["n_iters"],
        test_config["likelihood_fn"],
        prospective=prospective,
        critical_pct=test_config["critical_pct"],
        append_results=append_results,
    )

    logging.info("S-Test {}".format(test_results["test_res"]))
    logging.info("S-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("S-Test model pct: {}".format(test_results["percentile"]))
    return test_results


def N_test_old(
    cfg: dict,
    bin_gdf: Optional[GeoDataFrame] = None,
) -> dict:
    """
    Tests

    """
    logging.info("Running N-Test")
    test_config = cfg["config"]["model_framework"]["relm"]["N_test"]

    if "prospective" not in test_config.keys():
        prospective = False
    else:
        prospective = test_config["prospective"]

    if "conf_interval" not in test_config:
        test_config["conf_interval"] = 0.95

    annual_rup_rate = get_model_annual_eq_rate(bin_gdf)
    obs_eqs = get_total_obs_eqs(bin_gdf, prospective)
    n_obs = len(obs_eqs)

    test_rup_rate = annual_rup_rate * test_config["investigation_time"]

    if test_config["prob_model"] == "poisson":
        test_result = N_test_poisson(
            n_obs, test_rup_rate, test_config["conf_interval"]
        )

    elif test_config["prob_model"] == "neg_binom":
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

    else:
        raise ValueError(
            f"{test_config['prob_model']} not a valid probability model"
        )

    logging.info(
        "N-Test number obs eqs: {}".format(test_result["n_obs_earthquakes"])
    )
    logging.info(
        "N-Test number pred eqs: {}".format(test_result["inv_time_rate"])
    )
    logging.info("N-Test {}".format(test_result["test_pass"]))

    return test_result


relm_test_dict = {
    "L_test": L_test,
    "N_test": N_test,
    "M_test": M_test,
    "S_test": S_test,
}
