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

# from openquake.hme.utils.plots import plot_mfd
# from openquake.hme.utils.stats import poisson_likelihood, poisson_log_likelihood
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    # N_test_poisson,
    # N_test_neg_binom,
    s_test_function,
    # subdivide_observed_eqs,
    # get_model_annual_eq_rate,
    # get_total_obs_eqs,
    # get_model_mfd,
    # get_obs_mfd,
    # s_test_bin,
    # s_test_gdf_series,
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


relm_test_dict = {
    "L_test": L_test,
    "N_test": N_test,
    "M_test": M_test,
    "S_test": S_test,
}
