import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

from openquake.hme.utils.io.source_processing import (
    make_cell_gdf_from_ruptures,
)


def process_results(cfg, input_data, results):

    test_cfg = cfg["config"]

    cell_gdf = make_cell_gdf_from_ruptures(input_data["rupture_gdf"])

    if "gem" in test_cfg["model_framework"]:
        if "S_test" in test_cfg["model_framework"]["gem"]:
            add_s_test_fracs_to_cell_gdf(
                results["gem"]["S_test"],
                cell_gdf,
                model_test_framework="gem",
            )
            add_s_test_rates_to_cell_gdf(
                results["gem"]["S_test"],
                cell_gdf,
                model_test_framework="gem",
            )

        if "moment_over_under" in test_cfg["model_framework"]["gem"]:
            add_moment_over_under_results_to_cell_gdf(
                results["gem"]["moment_over_under"], cell_gdf
            )

    if "relm" in test_cfg["model_framework"]:
        if "S_test" in test_cfg["model_framework"]["relm"]:
            add_s_test_fracs_to_cell_gdf(
                results["relm"]["S_test"],
                cell_gdf,
                model_test_framework="relm",
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


def add_s_test_rates_to_cell_gdf(
    s_test_results, cell_gdf, model_test_framework="gem"
):
    cell_ids = sorted(
        s_test_results["val"]["test_data"]["cell_loglikes"].keys()
    )

    mod_mfd = pd.Series(
        {
            c: s_test_results["val"]["test_data"]["cell_loglikes"][c][
                "mod_mfd"
            ]
            for c in cell_ids
        },
        name=f"{model_test_framework}_S_test_model_rate",
    )
    obs_mfd = pd.Series(
        {
            c: s_test_results["val"]["test_data"]["cell_loglikes"][c][
                "obs_mfd"
            ]
            for c in cell_ids
        },
        name=f"{model_test_framework}_S_test_observed_rate",
    )

    mod_rates = pd.Series(
        {
            c: s_test_results["val"]["test_data"]["cell_loglikes"][c][
                "mod_rate"
            ]
            for c in cell_ids
        },
        name=f"{model_test_framework}_S_test_model_rate",
    )
    obs_rates = pd.Series(
        {
            c: s_test_results["val"]["test_data"]["cell_loglikes"][c][
                "obs_rate"
            ]
            for c in cell_ids
        },
        name=f"{model_test_framework}_S_test_observed_rate",
    )
    cell_gdf[f"{model_test_framework}_S_test_model_mfd"] = mod_mfd
    cell_gdf[f"{model_test_framework}_S_test_observed_mfd"] = obs_mfd
    cell_gdf[f"{model_test_framework}_S_test_model_rate"] = mod_rates
    cell_gdf[f"{model_test_framework}_S_test_observed_eqs"] = obs_rates

    return


def add_l_test_fracs_to_cell_gdf(
    l_test_results, cell_gdf, model_test_framework="gem"
):
    raise NotImplementedError


def add_moment_over_under_results_to_cell_gdf(mo_ov_un_results, cell_gdf):

    cell_ids = cell_gdf.index

    fracs = pd.Series(
        mo_ov_un_results["val"]["test_data"]["cell_fracs"],
        # {c: mo_ov_un_results["val"]["test_data"]["cell_fracs"][i]
        # for i, c in enumerate(cell_ids)},
        name="moment_over_under_frac",
    )

    cell_gdf["moment_over_under_frac"] = fracs

    rats = pd.Series(
        {
            c: mo_ov_un_results["val"]["test_data"]["obs_cell_moments"][c]
            / mo_ov_un_results["val"]["test_data"]["cell_model_moments"][c]
            for c in cell_ids
        },
        name="moment_ratio",
    )

    cell_gdf["moment_over_under_ratio"] = rats
