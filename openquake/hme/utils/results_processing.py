import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

from openquake.hme.utils.io.source_processing import make_cell_gdf_from_ruptures


def process_results(cfg, input_data, results):

    test_cfg = cfg["config"]

    cell_gdf = make_cell_gdf_from_ruptures(input_data["rupture_gdf"])

    if "gem" in test_cfg["model_framework"]:
        if "S_test" in test_cfg["model_framework"]["gem"]:
            add_s_test_fracs_to_cell_gdf(
                results["gem"]["S_test"], cell_gdf, model_test_framework="gem"
            )

    if "relm" in test_cfg["model_framework"]:
        if "S_test" in test_cfg["model_framework"]["relm"]:
            add_s_test_fracs_to_cell_gdf(
                results["relm"]["S_test"], cell_gdf, model_test_framework="relm"
            )

    results["cell_gdf"] = cell_gdf

    return


def add_s_test_fracs_to_cell_gdf(
    s_test_results, cell_gdf, model_test_framework="gem"
):
    cell_ids = sorted(
        s_test_results["val"]["test_data"]["cell_loglikes"].keys()
    )

    fracs = pd.Series(
        {
            c: s_test_results["val"]["test_data"]["cell_fracs"][i]
            for i, c in enumerate(cell_ids)
        },
        name=f"{model_test_framework}_S_test_frac",
    )

    likes = pd.Series(
        {
            c: np.mean(
                s_test_results["val"]["test_data"]["cell_loglikes"][c][
                    "stoch_loglikes"
                ]
            )
            for c in cell_ids
        },
        name=f"{model_test_framework}_S_test_log_likelihood",
    )

    cell_gdf[f"{model_test_framework}_S_test_frac"] = fracs
    cell_gdf[f"{model_test_framework}_S_test_log_like"] = likes

    return
