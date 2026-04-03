"""
Utility functions for running tests in the GEM model test framework.
"""
import h3
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from multiprocessing import Pool

from openquake.hazardlib.geo.geodetic import distance

from openquake.hme.utils import (
    mag_to_mo,
    sample_rups,
    get_model_mfd,
    get_obs_mfd,
    strike_dip_to_norm_vec,
    angles_between_plane_and_planes,
    angles_between_rake_and_rakes,
)
from openquake.hme.utils.utils import _n_procs
from openquake.hme.utils.stats import weighted_geom_mean


def get_rupture_gdf_cell_moment(rupture_gdf, t_yrs, rup_groups=None):
    """Computes the expected seismic moment per spatial cell and total, given
    rupture occurrence rates scaled by duration.
    
    :param rupture_gdf: GeoDataFrame of ruptures with ``magnitude``,
        ``occurrence_rate``, and ``cell_id`` columns.
    :param t_yrs: Duration in years to scale occurrence rates.
    :param rup_groups: Optional pre-computed groupby on ``cell_id``.
    :returns: Tuple of (per-cell moment Series, total moment float).
    """
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
    """Computes the total seismic moment per spatial cell and overall from an
    earthquake catalog.
    
    :param eq_df: GeoDataFrame of earthquakes with ``magnitude`` and
        ``cell_id`` columns.
    :param eq_groups: Optional pre-computed groupby on ``cell_id``.
    :returns: Tuple of (per-cell moment dict, total moment float).
    """
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
    """Compares observed seismic moment release to stochastic moment release
    from the model, per cell and in total.
    
    Generates ``n_iters`` stochastic catalogs by sampling ruptures, computes
    moment release for each, and calculates the fractile of the observed
    moment within the stochastic distribution.
    
    :param rup_df: GeoDataFrame of ruptures.
    :param eq_gdf: GeoDataFrame of observed earthquakes.
    :param cell_groups: Pre-computed groupby of ruptures on ``cell_id``.
    :param t_yrs: Duration in years.
    :param min_mag: Minimum magnitude for moment calculation.
    :param max_mag: Maximum magnitude for moment calculation.
    :param n_iters: Number of stochastic catalogs to generate.
    :returns: Dict with ``test_data`` containing per-cell and total moment
        comparisons and fractiles.
    """
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


def model_mfd_eval_fn(
    rup_gdf,
    eq_gdf,
    mag_bins,
    t_yrs=None,
    completeness_table=None,
    annualize=False,
    stop_date=None,
):

    """Computes and compares model and observed magnitude-frequency
    distributions.
    
    :param rup_gdf: GeoDataFrame of ruptures.
    :param eq_gdf: GeoDataFrame of observed earthquakes.
    :param mag_bins: Dict of magnitude bin centers to (min, max) tuples.
    :param t_yrs: Duration in years.
    :param completeness_table: Optional completeness table as list of
        [year, magnitude] pairs.
    :param annualize: If True, annualize rates.
    :param stop_date: End date of the catalog.
    :returns: Dict with ``test_data`` containing a DataFrame of model and
        observed MFDs (incremental and cumulative).
    """
    if annualize:
        t_yrs_model = 1.0
        completeness_table_model = None

    else:
        completeness_table_model = completeness_table
        t_yrs_model = t_yrs

    mod_mfd = get_model_mfd(
        rup_gdf,
        mag_bins,
        cumulative=False,
        t_yrs=t_yrs_model,
        completeness_table=completeness_table_model,
        stop_date=stop_date,
    )
    obs_mfd = get_obs_mfd(
        eq_gdf,
        mag_bins,
        t_yrs=t_yrs,
        cumulative=False,
        completeness_table=completeness_table,
        annualize=annualize,
        stop_date=stop_date,
    )

    mfd_df = pd.DataFrame.from_dict(
        mod_mfd, orient="index", columns=["mod_mfd"]
    )

    mfd_df["mod_mfd_cum"] = np.cumsum(mfd_df["mod_mfd"].values[::-1])[::-1]

    mfd_df["obs_mfd"] = obs_mfd.values()
    mfd_df["obs_mfd_cum"] = np.cumsum(mfd_df["obs_mfd"].values[::-1])[::-1]

    mfd_df.index.name = "bin"

    return {"test_data": {"mfd_df": mfd_df, "annualize": annualize}}


def get_moment_from_mfd(mfd: dict) -> float:
    """Calculates total seismic moment from an MFD dictionary.
    
    :param mfd: Dict mapping magnitude bin centers to rates.
    :returns: Total seismic moment (N*m).
    """
    if isinstance(mfd, dict):
        return _get_moment_from_mfd_dict(mfd)
    else:
        raise ValueError("Only dict mfd currently supported")


def _get_moment_from_mfd_dict(mfd: dict) -> float:
    mo = sum(
        mag_to_mo(np.array(list(mfd.keys()))) * np.array(list(mfd.values()))
    )

    return mo


def mag_diff_likelihood(eq_mag, rup_mags, mag_window=1.0):
    """Calculates a linear likelihood based on the magnitude difference
    between an earthquake and candidate ruptures.
    
    :param eq_mag: Observed earthquake magnitude.
    :param rup_mags: Array of rupture magnitudes.
    :param mag_window: Total width of the magnitude window for matching.
    :returns: Array of likelihoods in [0, 1], where 1 means exact match.
    """
    likes = 1 - np.abs(eq_mag - rup_mags) / (mag_window / 2.0)
    if np.isscalar(likes):
        if likes < 0.0:
            likes = 0.0
    else:
        likes[likes < 0.0] = 0.0

    return likes


def get_distances(eq, rup_gdf):
    # this assumes we want 3d distance instead of separate treatment
    # of h, v dists
    """Calculates 3D distances between an earthquake and a set of ruptures.
    
    :param eq: Earthquake row with ``longitude``, ``latitude``, ``depth``.
    :param rup_gdf: GeoDataFrame of ruptures with the same columns.
    :returns: Array of distances in km.
    """
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
    """Filters ruptures to those within a magnitude window of the earthquake.
    
    :param eq: Earthquake row with ``magnitude``.
    :param rup_df: DataFrame of ruptures with ``magnitude`` column.
    :param mag_window: Total width of the magnitude window.
    :returns: Filtered DataFrame of ruptures within the window.
    """
    rdf_lo = rup_df.loc[
        rup_df.magnitude.values <= (eq.magnitude + mag_window / 2.0)
    ]
    rdf_in_range = rdf_lo.loc[
        rdf_lo.magnitude.values >= eq.magnitude - mag_window / 2.0
    ]

    return rdf_in_range


def get_nearby_rups(eq, rup_df):
    # first find adjacent cells to pare down search space
    """Finds ruptures in the earthquake's H3 cell and its immediate neighbors.
    
    :param eq: Earthquake row with ``cell_id``.
    :param rup_df: DataFrame of ruptures with ``cell_id`` column.
    :returns: Filtered DataFrame of nearby ruptures.
    """
    closest_cells = h3.grid_disk(eq.cell_id, 1)

    rups_nearby = rup_df.loc[rup_df.cell_id.isin(closest_cells)]

    return rups_nearby


def get_matching_rups(
    eq,
    rup_gdf,
    distance_lambda=2.0,
    dist_by_mag=True,
    mag_window=1.0,
    group_return_threshold=0.9,
    min_likelihood=0.1,
    no_attitude_default_like=0.5,
    no_rake_default_like=0.5,
    use_occurrence_rate=False,
    return_one=False,
    attitude_rel_weight=0.25,
    rake_rel_weight=0.25,
    mag_rel_weight=1.0,
):
    # selection phase
    """Finds and ranks modeled ruptures that match an observed earthquake.
    
    Matching is done in two phases: selection (nearby ruptures within a
    magnitude window) and ranking (weighted geometric mean of distance,
    magnitude, attitude, and rake likelihoods). If focal mechanism data is
    available (single or double-couple), attitude and rake similarity are
    included; otherwise, default likelihoods are used.
    
    :param eq: Earthquake row with location, magnitude, and optional focal
        mechanism columns (``strike``, ``dip``, ``rake`` or
        ``strike1``/``strike2`` etc.).
    :param rup_gdf: GeoDataFrame of candidate ruptures.
    :param distance_lambda: Distance decay parameter (scaled by magnitude
        if ``dist_by_mag`` is True).
    :param dist_by_mag: Scale distance decay by earthquake magnitude.
    :param mag_window: Total width of the magnitude window for candidates.
    :param group_return_threshold: Fraction of max likelihood below which
        matches are discarded.
    :param min_likelihood: Absolute minimum likelihood for a match.
    :param no_attitude_default_like: Default attitude likelihood when no
        focal mechanism is available.
    :param no_rake_default_like: Default rake likelihood when no focal
        mechanism is available.
    :param use_occurrence_rate: Include occurrence rate in the ranking.
    :param return_one: ``False`` to return all matches, ``"best"`` for the
        top match, ``"sample"`` to sample weighted by likelihood.
    :param attitude_rel_weight: Relative weight for attitude similarity.
    :param rake_rel_weight: Relative weight for rake similarity.
    :param mag_rel_weight: Relative weight for magnitude similarity.
    :returns: DataFrame of matching ruptures (or Series if ``return_one``),
        or ``None`` if no matches found.
    """
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
    mag_likes[mag_likes < 1e-20] = 1e-20
    rups["mag_like"] = np.float64(mag_likes)

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
        rake_diffs = pd.Series(rake_diffs, index=rups.index)
        # angles > pi/2 should all have zero likelihood
        rake_likes = np.cos(rake_diffs)
        rake_likes[rake_likes < 1e-20] = 1e-20
        rups["rake_diff"] = rake_diffs

    elif hasattr(eq, "strike1") and not np.isnan(eq.strike1):
        # plane attitude diffs for first plane
        attitude_diffs1 = angles_between_plane_and_planes(
            eq.strike1,
            eq.dip1,
            rups.strike.values,
            rups.dip.values,
            return_radians=True,
        )
        attitude_diffs1 = pd.Series(attitude_diffs1, index=rups.index)
        attitude_likes1 = np.cos(attitude_diffs1)
        attitude_likes1[attitude_likes1 < 1e-20] = 1e-20

        # rake diffs for first plane
        rake_diffs1 = angles_between_rake_and_rakes(
            eq.rake1, rups.rake, return_radians=True
        )
        rake_likes1 = np.cos(rake_diffs1)
        rake_likes1[rake_likes1 < 1e-20] = 1e-20

        # Total likelihood for first plane (multiply attitude and rake likelihoods)
        total_likes1 = attitude_likes1 * rake_likes1

        # plane attitude diffs for second plane
        attitude_diffs2 = angles_between_plane_and_planes(
            eq.strike2,
            eq.dip2,
            rups.strike.values,
            rups.dip.values,
            return_radians=True,
        )
        attitude_diffs2 = pd.Series(attitude_diffs2, index=rups.index)
        attitude_likes2 = np.cos(attitude_diffs2)
        attitude_likes2[attitude_likes2 < 1e-20] = 1e-20

        # rake diffs for second plane
        rake_diffs2 = angles_between_rake_and_rakes(
            eq.rake2, rups.rake, return_radians=True
        )
        rake_likes2 = np.cos(rake_diffs2)
        rake_likes2[rake_likes2 < 1e-20] = 1e-20

        # Total likelihood for second plane
        total_likes2 = attitude_likes2 * rake_likes2

        # Create boolean mask for where plane 1 is more likely
        plane1_more_likely = total_likes1 > total_likes2

        # Use the mask to select the appropriate differences
        attitude_diffs = np.where(
            plane1_more_likely, attitude_diffs1, attitude_diffs2
        )
        rake_diffs = np.where(plane1_more_likely, rake_diffs1, rake_diffs2)

        rups["attitude_diff"] = attitude_diffs
        rups["rake_diff"] = rake_diffs
        attitude_likes = np.cos(attitude_diffs)
        attitude_likes[attitude_likes < no_attitude_default_like] = (
            no_attitude_default_like
        )
        rake_likes = np.cos(rake_diffs)
        rake_likes[rake_likes < no_attitude_default_like] = (
            no_attitude_default_like
        )

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
        total_likes = weighted_geom_mean(
            dist_likes,
            mag_likes,
            attitude_likes,
            rake_likes,
            rates_norm,
            weights=np.array(
                [
                    1.0,
                    mag_rel_weight,
                    attitude_rel_weight,
                    rake_rel_weight,
                    1.0,
                ]
            ),
        )
    else:
        total_likes = weighted_geom_mean(
            dist_likes,
            mag_likes,
            attitude_likes,
            rake_likes,
            weights=np.array(
                [1.0, mag_rel_weight, attitude_rel_weight, rake_rel_weight]
            ),
        )

    rups["likelihood"] = total_likes
    rups = rups.sort_values("likelihood", ascending=False)
    max_like = total_likes.max()

    rups = rups.loc[rups.likelihood >= max_like * group_return_threshold]
    rups = rups.loc[rups.likelihood >= min_likelihood]

    rups["eq"] = eq.name

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
    """Matches all earthquakes in a catalog to their best-matching modeled
    ruptures using :func:`get_matching_rups`.
    
    :param eq_gdf: GeoDataFrame of observed earthquakes.
    :param rup_gdf: GeoDataFrame of modeled ruptures.
    :param distance_lambda: Distance decay parameter.
    :param dist_by_mag: Scale distance decay by earthquake magnitude.
    :param mag_window: Magnitude window for candidate ruptures.
    :param group_return_threshold: Fraction of max likelihood threshold.
    :param no_attitude_default_like: Default attitude likelihood.
    :param no_rake_default_like: Default rake likelihood.
    :param use_occurrence_rate: Include occurrence rate in ranking.
    :param return_one: ``"best"``, ``"sample"``, or ``False``.
    :param parallel: Use multiprocessing (currently disabled).
    :returns: List of match results (DataFrames or None) per earthquake.
    """
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
    if False is True:
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
    """Runs the rupture matching evaluation, matching all observed earthquakes
    to modeled ruptures and collecting matched/unmatched results.
    
    :param rup_gdf: GeoDataFrame of modeled ruptures.
    :param eq_gdf: GeoDataFrame of observed earthquakes.
    :returns: Dict with ``matched_rups`` DataFrame and ``unmatched_eqs``
        DataFrame.
    """
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
        "mag_like",
    ]:
        matched_rups[col] = matched_rups[col].astype(float)

    matched_rups["rupture"] = matched_rups.index.values
    matched_rups["eq"] = matched_indices

    matched_rups.set_index("eq", inplace=True)
    unmatched_eqs = eq_gdf.loc[unmatched_indices]

    return {"matched_rups": matched_rups, "unmatched_eqs": unmatched_eqs}


def get_closest_rupture(eq, rupture_df):
    """Returns the rupture closest to the given earthquake in 3D distance."""
    dists = get_distances(eq, rupture_df)
    return rupture_df.iloc[dists.argmin()]
