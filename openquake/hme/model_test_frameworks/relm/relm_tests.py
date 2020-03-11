import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils.stats import (
    negative_binomial_distribution,
    estimate_negative_binom_parameters,
)
from openquake.hme.utils import get_source_bins
from openquake.hme.utils.plots import plot_mfd
from openquake.hme.utils.stats import poisson_likelihood, poisson_log_likelihood
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    N_test_poisson,
    N_test_neg_binom,
    subdivide_observed_eqs,
)


def L_test():
    #
    raise NotImplementedError


def M_test(cfg, bin_gdf: Optional[GeoDataFrame] = None,) -> dict:

    """

    """
    logging.info("Running CSEP/RELM M-Test")

    test_config = cfg["config"]["model_framework"]["relm"]["M_test"]

    if "prospective" not in test_config.keys():
        prospective = False
    else:
        prospective = test_config["prospective"]

    if "conf_interval" not in test_config:
        test_config["conf_interval"] = 0.25

    ##
    # get model and observed MFDs
    ##
    mod_mfd = bin_gdf.iloc[0].SpacemagBin.get_rupture_mfd()

    if prospective is False:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_empirical_mfd(
            t_yrs=test_config["investigation_time"]
        )
    else:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_prospective_mfd(
            t_yrs=test_config["investigation_time"]
        )

    for i, row in bin_gdf.iloc[1:].iterrows():
        bin_mod_mfd = row.SpacemagBin.get_rupture_mfd()

        if prospective is False:
            bin_obs_mfd = row.SpacemagBin.get_empirical_mfd(
                t_yrs=test_config["investigation_time"]
            )
        else:
            bin_obs_mfd = row.SpacemagBin.get_prospective_mfd(
                t_yrs=test_config["investigation_time"]
            )

        for bin_center, rate in bin_mod_mfd.items():
            mod_mfd[bin_center] += rate

        for bin_center, rate in bin_obs_mfd.items():
            obs_mfd[bin_center] += rate

    n_bins = len(mod_mfd.keys())

    stochastic_eq_counts = {
        bc: np.random.poisson(
            (rate * test_config["investigation_time"]), size=test_config["n_iters"]
        )
        for bc, rate in mod_mfd.items()
    }

    bin_log_likelihoods = {
        bc: [
            poisson_log_likelihood(
                n_stoch, (mod_mfd[bc] * test_config["investigation_time"])
            )
            for n_stoch in eq_counts
        ]
        for bc, eq_counts in stochastic_eq_counts.items()
    }

    stoch_geom_mean_likes = np.array(
        [
            np.exp(
                np.sum([bll_mag[i] for bll_mag in bin_log_likelihoods.values()])
                / n_bins
            )
            for i in range(test_config["n_iters"])
        ]
    )

    obs_geom_mean_like = np.exp(
        np.sum(
            [
                poisson_log_likelihood(
                    int(obs_mfd[bc] * test_config["investigation_time"]),
                    rate * test_config["investigation_time"],
                )
                for bc, rate in mod_mfd.items()
            ]
        )
        / n_bins
    )

    pctile = len(stoch_geom_mean_likes < obs_geom_mean_like) / test_config["n_iters"]

    test_pass = True if pctile >= test_config["conf_interval"] else False
    test_res = "Pass" if test_pass else "Fail"

    test_result = {
        "conf_interval": test_config["conf_interval"],
        "percentile": pctile,
        "test_pass": test_pass,
        "test_res": test_res,
    }

    logging.info("M-Test {}".format(test_res))
    return test_result


def S_test():
    """
    """

    pass


def N_test(cfg: dict, bin_gdf: Optional[GeoDataFrame] = None,) -> dict:
    """

    """

    logging.info("Running N-Test")
    test_config = cfg["config"]["model_framework"]["relm"]["N_test"]

    if "prospective" not in test_config.keys():
        prospective = False
    else:
        prospective = test_config["prospective"]

    if "conf_interval" not in test_config:
        test_config["conf_interval"] = 0.95

    annual_rup_rate = 0.0
    obs_eqs = []
    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin
        min_bin_center = np.min(sb.mag_bin_centers)
        bin_mfd = sb.get_rupture_mfd(cumulative=True)
        annual_rup_rate += bin_mfd[min_bin_center]

        if prospective is False:
            for mb in sb.observed_earthquakes.values():
                obs_eqs.extend(mb)
        else:
            for mb in sb.prospective_earthquakes.values():
                obs_eqs.extend(mb)

    test_rup_rate = annual_rup_rate * test_config["investigation_time"]

    if test_config["prob_model"] == "poisson":
        test_result = N_test_poisson(
            len(obs_eqs), test_rup_rate, test_config["conf_interval"]
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
            len(obs_eqs),
            test_rup_rate,
            prob_success,
            r_dispersion,
            test_config["conf_interval"],
        )

    else:
        raise ValueError(f"{test_config['prob_model']} not a valid probability model")

    return test_result


relm_test_dict = {"L_test": L_test, "N_test": N_test, "M_test": M_test}
