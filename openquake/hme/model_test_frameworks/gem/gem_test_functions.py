"""
Utility functions for running tests in the GEM model test framework.
"""

from multiprocessing import Pool

from h3 import h3
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from tqdm.autonotebook import tqdm

from openquake.hazardlib.geo.geodetic import distance

from openquake.hme.utils import (
    parallelize,
    mag_to_mo,
    sample_rups,
    get_model_mfd,
    get_obs_mfd,
    strike_dip_to_norm_vec,
    angles_between_plane_and_planes,
    angles_between_rake_and_rakes,
)
from openquake.hme.utils.utils import _n_procs
from openquake.hme.utils.stats import geom_mean


def get_rupture_gdf_cell_moment(rupture_gdf, t_yrs, rup_groups=None):
    if rup_groups == None:
        rup_groups = rupture_gdf.groupby("cell_id")

    moment_sums = pd.Series(
        {
            name: (
                mag_to_mo(group["magnitude"])
                * group["occurrence_rate"]
                * t_yrs
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


def mag_diff_likelihood(eq_mag, rup_mags, mag_window=1.0):
    return 1 - np.abs(eq_mag - rup_mags)  # / (mag_window / 2.)


def get_distances(eq, rup_gdf):
    # this assumes we want 3d distance instead of separate treatment
    # of h, v dists
    dists = distance(
        eq.longitude,
        eq.latitude,
        eq.depth,
        rup_gdf["longitude"],
        rup_gdf["latitude"],
        rup_gdf["depth"],
    )
    return dists


def get_rups_in_mag_range(eq, rup_df, mag_window=1.0):
    rdf_lo = rup_df.loc[
        rup_df.magnitude.values <= (eq.magnitude + mag_window / 2.0)
    ]
    rdf_in_range = rdf_lo.loc[
        rdf_lo.magnitude.values >= eq.magnitude - mag_window / 2.0
    ]

    return rdf_in_range


def get_nearby_rups(eq, rup_df):
    # first find adjacent cells to pare down search space
    closest_cells = h3.k_ring(eq.cell_id, 1)

    rups_nearby = rup_df.loc[rup_df.cell_id.isin(closest_cells)]

    return rups_nearby


def get_matching_rups(
    eq,
    rup_gdf,
    distance_lambda=1.0,
    dist_by_mag=True,
    mag_window=1.0,
    group_return_threshold=0.9,
    min_likelihood=0.1,
    no_attitude_default_like=0.5,
    no_rake_default_like=0.5,
    use_occurrence_rate=False,
    return_one=False,
):
    # selection phase
    rups = get_nearby_rups(eq, rup_df=rup_gdf)
    rups = get_rups_in_mag_range(eq, rup_df=rups, mag_window=mag_window)

    # ranking phase

    # distances
    if dist_by_mag:
        dist_constant = distance_lambda * eq.magnitude
    else:
        dist_constant = distance_lambda
    dists = get_distances(eq, rups)
    rups = rups[dists <= dists.min() * dist_constant]
    dists = dists[dists <= dists.min() * dist_constant]
    dist_likes = np.exp(-dists / dist_constant)

    rups = rups[dist_likes >= 0.0]  # a lil more filtering, to speed things up
    dists = dists[dist_likes >= 0.0]

    rups["eq_dist"] = dists

    # magnitudes
    mag_likes = mag_diff_likelihood(
        eq.magnitude, rups.magnitude, mag_window=mag_window
    )

    if hasattr(eq, "strike") and not np.isnan(eq.strike):
        # plane attitude diffs
        attitude_diffs = angles_between_plane_and_planes(
            eq.strike,
            eq.dip,
            rups.strike.values,
            rups.dip.values,
            return_radians=True,
        )
        attitude_diffs = pd.Series(attitude_diffs, index=rups.index)
        attitude_likes = np.cos(attitude_diffs)
        attitude_likes[attitude_likes <= 0.0] = 1e-20
        rups["attitude_diff"] = attitude_diffs

        # rakes
        rake_diffs = angles_between_rake_and_rakes(
            eq.rake, rups.rake, return_radians=True
        )
        # angles > pi/2 should all have zero likelihood
        # rake_diffs[rake_diffs >= np.pi / 2] = np.pi / 2
        rake_likes = np.cos(rake_diffs)
        rake_likes[rake_likes <= 0.0] = 1e-20
        rups["rake_diff"] = rake_diffs
    else:
        attitude_likes = np.ones(len(rups)) * no_attitude_default_like
        rups["attitude_diff"] = np.empty(len(rups))
        rups["attitude_diff"].values[:] = np.nan

        rake_likes = np.ones(len(rups)) * no_rake_default_like
        rups["rake_diff"] = np.empty(len(rups))
        rups["rake_diff"].values[:] = np.nan

    # put it all together
    if use_occurrence_rate:
        rates_norm = rups.occurrence_rate / rups.occurrence_rate.max()
        total_likes = geom_mean(
            dist_likes, mag_likes, attitude_likes, rake_likes, rates_norm
        )
    else:
        total_likes = geom_mean(
            dist_likes,
            mag_likes,
            attitude_likes,
            rake_likes,
        )

    rups["likelihood"] = total_likes
    rups = rups.sort_values("likelihood", ascending=False)
    max_like = total_likes.max()

    rups = rups.loc[rups.likelihood >= max_like * group_return_threshold]
    rups = rups.loc[rups.likelihood >= min_likelihood]

    if len(rups) == 0:
        return None

    if return_one is False:
        return rups
    elif return_one == "best":
        return rups.iloc[0]
    elif return_one == "sample":
        weights = rups.likelihood.values / sum(rups.likelihood.values)
        idx = np.random.choice(rups.index.values, p=weights)
        return rups.loc[idx]
    else:
        raise ValueError(
            "Choose False, 'best', or 'sample' for return_one. "
            + f"(current value is {return_one}"
        )


def _get_matching_rups(args):
    eq = args[0]
    rup_gdf = args[1]
    distance_lambda = args[2]
    dist_by_mag = args[3]
    mag_window = args[4]
    group_return_threshold = args[5]
    no_attitude_default_like = args[6]
    no_rake_default_like = args[7]
    use_occurrence_rate = args[8]
    return_one = args[9]

    return get_matching_rups(
        eq,
        rup_gdf,
        distance_lambda=distance_lambda,
        dist_by_mag=dist_by_mag,
        mag_window=mag_window,
        group_return_threshold=group_return_threshold,
        no_attitude_default_like=no_attitude_default_like,
        no_rake_default_like=no_rake_default_like,
        use_occurrence_rate=use_occurrence_rate,
        return_one=return_one,
    )


def match_eqs_to_rups(
    eq_gdf,
    rup_gdf,
    distance_lambda=1.0,
    dist_by_mag=True,
    mag_window=1.0,
    group_return_threshold=0.9,
    no_attitude_default_like=0.5,
    no_rake_default_like=0.5,
    use_occurrence_rate=False,
    return_one="best",
    parallel=False,
):
    match_rup_args = (
        (
            eq,
            rup_gdf,
            distance_lambda,
            dist_by_mag,
            mag_window,
            group_return_threshold,
            no_attitude_default_like,
            no_rake_default_like,
            use_occurrence_rate,
            return_one,
        )
        for i, eq in eq_gdf.iterrows()
    )
    if parallel is True:
        with Pool(_n_procs) as pool:
            match_results = list(
                tqdm(
                    pool.imap(_get_matching_rups, match_rup_args, chunksize=10)
                )
            )
            _ = len(match_results)

    else:
        match_results = [
            _get_matching_rups(arg)
            for arg in tqdm(match_rup_args, total=len(eq_gdf))
        ]

    return match_results


def rupture_matching_eval_fn(
    rup_gdf,
    eq_gdf,
    distance_lambda=1.0,
    dist_by_mag=True,
    mag_window=1.0,
    group_return_threshold=0.9,
    no_attitude_default_like=0.5,
    no_rake_default_like=0.5,
    use_occurrence_rate=False,
    return_one="best",
    parallel=False,
):
    match_results = match_eqs_to_rups(
        eq_gdf,
        rup_gdf,
        distance_lambda=distance_lambda,
        dist_by_mag=dist_by_mag,
        mag_window=mag_window,
        group_return_threshold=group_return_threshold,
        no_attitude_default_like=no_attitude_default_like,
        no_rake_default_like=no_rake_default_like,
        use_occurrence_rate=use_occurrence_rate,
        parallel=parallel,
        return_one=return_one,
    )

    matched_indices = []
    unmatched_indices = []
    matched_rup_list = []
    for i, match in enumerate(match_results):
        if match is not None:
            matched_rup_list.append(match)
            matched_indices.append(eq_gdf.index.values[i])
        else:
            unmatched_indices.append(eq_gdf.index.values[i])

    matched_rups = pd.concat(matched_rup_list, axis=1).T

    for col in [
        "longitude",
        "latitude",
        "depth",
        "magnitude",
        "strike",
        "dip",
        "rake",
        "occurrence_rate",
        "eq_dist",
        "attitude_diff",
        "rake_diff",
        "likelihood",
    ]:
        matched_rups[col] = matched_rups[col].astype(float)

    matched_rups["eq"] = matched_indices

    matched_rups.set_index("eq", inplace=True)
    unmatched_eqs = eq_gdf.loc[unmatched_indices]

    return {"matched_rups": matched_rups, "unmatched_eqs": unmatched_eqs}


def process_eq_match_results(match_results):
    pass
