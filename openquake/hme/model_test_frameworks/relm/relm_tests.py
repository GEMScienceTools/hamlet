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
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    N_test_poisson,
    N_test_neg_binom,
    subdivide_observed_eqs,
)


def L_test():
    #
    raise NotImplementedError


def N_test(
    cfg: dict,
    bin_gdf: Optional[GeoDataFrame] = None,
    obs_seis_catalog: Optional[GeoDataFrame] = None,
    pro_seis_catalog: Optional[GeoDataFrame] = None,
    validate: bool = False,
) -> dict:
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


relm_test_dict = {"L_test": L_test, "N_test": N_test}
