"""
Utility functions for running tests in the GEM model test framework.
"""
import logging
from typing import Sequence, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from geopandas import GeoSeries

from openquake.hme.utils import (
    # SpacemagBin,
    parallelize,
    mag_to_mo,
    sample_rups,
    get_model_mfd,
    get_obs_mfd,
)


def get_rupture_gdf_cell_moment(rupture_gdf, t_yrs, rup_groups=None):
    if rup_groups == None:
        rup_groups = rupture_gdf.groupby("cell_id")

    moment_sums = pd.Series(
        {
            name: (
                mag_to_mo(group["magnitude"]) * group["occurrence_rate"] * t_yrs
            ).sum()
            for name, group in rup_groups
        }
    )

    total_moment = moment_sums.sum()

    return moment_sums, total_moment


def get_catalog_moment(eq_df, eq_groups=None):
    if eq_groups == None:
        eq_groups = eq_df.groupby("cell_id")

    moment_sums = {
        name: mag_to_mo(group["magnitude"]).sum() for name, group in eq_groups
    }

    total_sum = sum(moment_sums.values())

    return moment_sums, total_sum


def moment_over_under_eval_fn(
    rup_df, eq_gdf, cell_groups, t_yrs, min_mag=1.0, max_mag=10.0, n_iters=1000
):
    cell_ids = sorted(rup_df.cell_id.unique())

    cell_model_moments, total_model_moment = get_rupture_gdf_cell_moment(
        rup_df, t_yrs, rup_groups=cell_groups
    )

    cell_moment_iterations = {
        cell_id: np.zeros(n_iters) for cell_id in cell_ids
    }

    total_moment_iterations = np.zeros(n_iters)

    iter_moments = {}

    for i in range(n_iters):
        rup_sample = sample_rups(
            rup_df, t_yrs, min_mag=min_mag, max_mag=max_mag
        )
        iter_moments[i], iter_moment_sum = get_catalog_moment(rup_sample)

        for cell_id, moment_sum in iter_moments[i].items():
            cell_moment_iterations[cell_id][i] += moment_sum

        total_moment_iterations[i] += iter_moment_sum

    cat_cell_moments, cat_total_moment = get_catalog_moment(eq_gdf)

    cat_cell_moments = pd.Series(
        index=cell_model_moments.index,
        data=np.zeros(
            len(
                cell_ids,
            )
        ),
    ).add(pd.Series(cat_cell_moments), fill_value=0.0)

    cell_fracs = {
        cell_id: sum(
            cell_moment_iterations[cell_id] < cat_cell_moments[cell_id]
        )
        / n_iters
        for cell_id in cell_ids
    }

    total_frac = sum(total_moment_iterations < cat_total_moment) / n_iters

    results = {
        "test_data": {
            "total_model_moment": total_model_moment,
            "cell_model_moments": cell_model_moments,
            "total_obs_moment": cat_total_moment,
            "modeled_obs_moment": {
                "mean": total_moment_iterations.mean(),
                "sd": np.std(total_moment_iterations),
            },
            "frac": total_frac,
            "cell_fracs": cell_fracs,
            "stoch_total_moments": total_moment_iterations,
            "stoch_cell_moments": iter_moments,
            "obs_cell_moments": cat_cell_moments,
            "model_moment_ratio": total_model_moment / cat_total_moment,
        }
    }

    return results


def model_mfd_eval_fn(rup_gdf, eq_gdf, mag_bins, t_yrs):
    mod_mfd = get_model_mfd(rup_gdf, mag_bins, cumulative=False)
    obs_mfd = get_obs_mfd(eq_gdf, mag_bins, t_yrs=t_yrs, cumulative=False)

    mfd_df = pd.DataFrame.from_dict(
        mod_mfd, orient="index", columns=["mod_mfd"]
    )

    mfd_df["mod_mfd_cum"] = np.cumsum(mfd_df["mod_mfd"].values[::-1])[::-1]

    mfd_df["obs_mfd"] = obs_mfd.values()
    mfd_df["obs_mfd_cum"] = np.cumsum(mfd_df["obs_mfd"].values[::-1])[::-1]

    mfd_df.index.name = "bin"

    return {"test_data": {"mfd_df": mfd_df}}


def get_moment_from_mfd(mfd: dict) -> float:
    mo = sum(
        mag_to_mo(np.array(list(mfd.keys()))) * np.array(list(mfd.values()))
    )

    return mo
