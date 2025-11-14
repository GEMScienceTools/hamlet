import logging

from openquake.hme.utils import (
    get_mag_bins_from_cfg,
)

from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    s_test_function,
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
    critical_frac = test_config.get("critical_frac", 0.25)

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
        critical_frac=critical_frac,
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
    logging.info("Running CSEP/RELM S-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["relm"]["S_test"]
    prospective = test_config.get("prospective", False)
    likelihood_function = test_config.get("likelihood_function", "mfd")
    normalize_n_eqs = test_config.get("normalize_n_eqs", False)
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
        mag_bins=mag_bins,
        normalize_n_eqs=normalize_n_eqs,
        completeness_table=completeness_table,
        stop_date=stop_date,
        critical_frac=test_config["critical_frac"],
        not_modeled_likelihood=not_modeled_likelihood,
        parallel=parallel,
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
        critical_frac=test_config["critical_frac"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("L-Test {}".format(test_results["test_res"]))
    logging.info("L-Test crit frac: {}".format(test_results["critical_frac"]))
    logging.info("L-Test model fractile: {}".format(test_results["fractile"]))
    return test_results


def N_test(cfg: dict, input_data: dict) -> dict:
    logging.info("Running N-Test")

    test_config = cfg["config"]["model_framework"]["relm"]["N_test"]
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


relm_test_dict = {
    "L_test": L_test,
    "N_test": N_test,
    "M_test": M_test,
    "S_test": S_test,
}
