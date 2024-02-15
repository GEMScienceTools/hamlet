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


def add_uncertainty_to_rups(rups):
    """
    Added by CB to assign uncertainty to rup strike, dip and rake
    """
    # Set strikes, dips and rakes with some uncertainty
    strike_plus, strike_minus = {}, {}
    dip_plus, dip_minus = {}, {}
    rake_plus, rake_minus = {}, {}
    
    strike_unc = 60
    dip_unc = 30
    rake_unc = 45    
    for row, rup in rups.iterrows():
    
        strike_plus_unc = rup.strike + strike_unc
        if strike_plus_unc > 360:
            strike_plus_unc = strike_plus_unc - 360
        strike_plus[row] = strike_plus_unc
        
        strike_minus_unc = rup.strike - strike_unc
        if strike_minus_unc < 0:
            strike_minus_unc = strike_minus_unc + 360
        strike_minus[row] = strike_minus_unc 
            
        dip_plus_unc = rup.dip + dip_unc
        if dip_plus_unc > 90:
            dip_plus_unc = 90
        dip_plus[row] = dip_plus_unc
        
        dip_minus_unc = rup.dip - dip_unc
        if dip_minus_unc < 0:
            dip_minus_unc = 0
        dip_minus[row] = dip_minus_unc
        
        rake_plus_unc = rup.rake + rake_unc
        rake_minus_unc = rup.rake - rake_unc
        if 0 <= rup.rake and rup.rake <= 180: # If initial rake positive
            if rake_plus_unc > 180:
                rake_plus_unc = -1*(rake_plus_unc) + 180
            if rake_minus_unc < 0:
                rake_minus_unc = -180 - rake_minus_unc
        
        if rup.rake >= -180 and rup.rake < 0: # If initial rake negative
            if rake_plus_unc >= 0:
                rake_plus_unc = 180 - rake_plus_unc
            if rake_minus_unc < -180:
                rake_minus_unc = -180 - rake_minus_unc
        
        rake_plus[row] = rake_plus_unc
        rake_minus[row] = rake_minus_unc
        
    rups['strike_plus_unc'] = pd.Series(strike_plus)
    rups['strike_minus_unc'] = pd.Series(strike_minus)
    rups['dip_plus_unc'] = pd.Series(dip_plus)
    rups['dip_minus_unc'] = pd.Series(dip_minus)
    rups['rake_plus_unc'] = pd.Series(rake_plus)
    rups['rake_minus_unc'] = pd.Series(rake_minus)
    
    return rups


def plane_evaluate(eq, rups):
    """
    Modified from original code to consider uncertainty in rup strike and dip
    and placed into seperate function.
    """
    strikes = [rups.strike_plus_unc, rups.strike, rups.strike_minus_unc]
    dips = [rups.dip_plus_unc, rups.dip, rups.dip_minus_unc]
    
    store_diffs, store_likes = [], []
    for strike_set in strikes:
        for dip_set in dips:
            rups_strikes = strike_set
            rups_dips = dip_set
        
            attitude_diffs = angles_between_plane_and_planes(
                eq.strike,
                eq.dip,
                rups_strikes,
                rups_dips,
                return_radians=True,
            )
            attitude_likes = np.cos(attitude_diffs)
            attitude_likes[attitude_likes <= 0.0] = 1e-20
            
            # Store per strike/dip combo
            store_diffs.append(attitude_diffs)
            store_likes.append(attitude_likes)
    
    # Get the diff and like per rup per strike/dip combo
    all_diffs_per_rup, all_likes_per_rup = [], []
    for idx_rup, rup in rups.iterrows():
        diffs_per_rup, likes_per_rup = [], []
        for idx_set, plane_set in enumerate(store_diffs):
            diffs_per_rup.append(plane_set[idx_rup])
            likes_per_rup.append(store_likes[idx_set][idx_rup])            
        all_diffs_per_rup.append(diffs_per_rup)
        all_likes_per_rup.append(likes_per_rup)
        
    # Per rup take the max likelihood and min diff
    max_like_store, min_diff_store = [], []
    for idx_rup, vals_per_rup in enumerate(all_likes_per_rup):
        # Max likelihood
        max_like_store.append(np.max(vals_per_rup))

        # Min diff
        diffs = all_diffs_per_rup[idx_rup]
        min_diff_idx = np.argmax(vals_per_rup)
        min_diff_store.append(diffs[min_diff_idx])
        
    # Use optimal likelihood and min diff
    attitude_diffs = np.array(min_diff_store).flatten()
    attitude_likes = np.array(max_like_store).flatten()
    
    return attitude_diffs, attitude_likes


def rake_evaluate(eq, rups):
    """
    Modified from original code to consider uncertainty in rup rake and placed
    into seperate function
    """
    ### rakes
    rakes = [rups.rake_plus_unc, rups.rake, rups.rake_minus_unc]
    store_rake_diffs, store_rake_likes = [], []
    for rake_set in rakes:
        rups_rakes = rake_set
        rake_diffs = angles_between_rake_and_rakes(
            eq.rake, rups_rakes, return_radians=True
        )
        # angles > pi/2 should all have zero likelihood
        # rake_diffs[rake_diffs >= np.pi / 2] = np.pi / 2
        rake_likes = np.cos(rake_diffs)
        rake_likes[rake_likes <= 0.0] = 1e-20
        
        # Store per rake set
        store_rake_diffs.append(rake_diffs)
        store_rake_likes.append(rake_likes)

    # Get the rake diff and rake like per rup per rup set
    all_rake_diffs_per_rup, all_rake_likes_per_rup = [], []
    for idx_rup, rup in rups.iterrows():
        rake_diffs_per_rup, rake_likes_per_rup = [], []
        for idx_set, rake_set in enumerate(store_rake_diffs):
            rake_diffs_per_rup.append(rake_set[idx_rup])
            rake_likes_per_rup.append(store_rake_likes[idx_set][idx_rup])            
        all_rake_diffs_per_rup.append(rake_diffs_per_rup)
        all_rake_likes_per_rup.append(rake_likes_per_rup)

    # Per rup take the max likelihood and min diff
    max_rake_like_store, min_rake_diff_store = [], []
    for idx_rup, vals_per_rup in enumerate(all_rake_likes_per_rup):
        # Max likelihood
        max_rake_like_store.append(np.max(vals_per_rup))

        # Min diff
        rake_diffs = all_rake_diffs_per_rup[idx_rup]
        rake_min_diff_idx = np.argmax(vals_per_rup)
        min_rake_diff_store.append(rake_diffs[rake_min_diff_idx])
        
    # Use optimal likelihood and min diff
    rake_diffs = np.array(min_rake_diff_store).flatten()
    rake_likes = np.array(max_rake_like_store).flatten()
    
    return rake_diffs, rake_likes
    

def get_matching_rups(
    eq,
    rup_gdf,
    distance_lambda=10.0,
    mag_window=1.0,
    return_threshold=0.9,
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
    dists = get_distances(eq, rups)
    rups = rups[dists <= dists.min() * distance_lambda]
    dists = dists[dists <= dists.min() * distance_lambda]
    dist_likes = np.exp(-dists / distance_lambda)

    rups = rups[dist_likes >= 0.0]  # a lil more filtering, to speed things up
    dists = dists[dist_likes >= 0.0]

    # The distances from the eq to the selected ruptures
    rups["eq_dist"] = dists

    # magnitudes (difference between eq mag and each rup mag)
    mag_likes = mag_diff_likelihood(
        eq.magnitude, rups.magnitude, mag_window=mag_window
    )
    if hasattr(eq, "strike") and not np.isnan(eq.strike) and \
        not np.isnan(eq.dip) and not np.isnan(eq.rake) and \
            len(rups.strike) != 0 and len(rups.dip) != 0:

        ### Modified here by CB
        rups = rups.reset_index()
        
        # Add uncertainty to rups strike, dip and rake
        rups = add_uncertainty_to_rups(rups)
        
        # Evaluate nodal planes
        attitude_diffs, attitude_likes = plane_evaluate(eq, rups)
        rups["attitude_diff"] = attitude_diffs
        
        # Evaluate rakes
        rake_diffs, rake_likes = rake_evaluate(eq, rups)
        rups["rake_diff"] = rake_diffs
        
    else:
        attitude_likes = np.ones(len(rups)) * no_attitude_default_like
        rups["attitude_diff"] = np.empty(len(rups))
        rups["attitude_diff"].values[:] = np.nan

        rake_likes = np.ones(len(rups)) * no_rake_default_like
        rups["rake_diff"] = np.empty(len(rups))
        rups["rake_diff"].values[:] = np.nan

    dist_likes = np.array(dist_likes)
    mag_likes = np.array(mag_likes)
    attitude_likes = np.array(attitude_likes)
    rake_likes = np.array(rake_likes)

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
    max_likes = total_likes.max()
    rups = rups.loc[rups.likelihood >= max_likes * return_threshold]

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

# Second function, called from within match_eqs_to_rups
def _get_matching_rups(args):
    eq = args[0]
    rup_gdf = args[1]
    distance_lambda = args[2]
    mag_window = args[3]
    return_threshold = args[4]
    no_attitude_default_like = args[5]
    no_rake_default_like = args[6]
    use_occurrence_rate = args[7]
    return_one = args[8]

    return get_matching_rups(
        eq,
        rup_gdf,
        distance_lambda=distance_lambda,
        mag_window=mag_window,
        return_threshold=return_threshold,
        no_attitude_default_like=no_attitude_default_like,
        no_rake_default_like=no_rake_default_like,
        use_occurrence_rate=use_occurrence_rate,
        return_one=return_one,
    )

# First function, called from rup_matching_eval
def match_eqs_to_rups(
    eq_gdf,
    rup_gdf,
    distance_lambda=10.0,
    mag_window=1.0,
    return_threshold=0.9,
    no_attitude_default_like=0.5,
    no_rake_default_like=0.5,
    use_occurrence_rate=False,
    return_one="best",
    parallel=False,
):
    # Get the information from each earthquake
    match_rup_args = (
        (
            eq,
            rup_gdf,
            distance_lambda,
            mag_window,
            return_threshold,
            no_attitude_default_like,
            no_rake_default_like,
            use_occurrence_rate,
            return_one,
        )
        for i, eq in eq_gdf.iterrows()
    )
    if parallel is True:
        # Get the arguments used in _get_matching_rups(arg)
        with Pool(_n_procs) as pool:
            match_results = list(
                tqdm(
                    pool.imap(_get_matching_rups, match_rup_args, chunksize=10)
                )
            )
            _ = len(match_results)

    else:
        # Get the arguments used in _get_matching_rups(arg)
        match_results = [
            _get_matching_rups(arg)
            for arg in tqdm(match_rup_args, total=len(eq_gdf))
        ]

    return match_results


# Begin with this function
def rup_matching_eval(
    eq_gdf,    # geo df
    rup_gdf,   # geo df
    distance_lambda=10.0,
    mag_window=1.0,
    return_threshold=0.9,
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
        mag_window=mag_window,
        return_threshold=return_threshold,
        no_attitude_default_like=no_attitude_default_like,
        no_rake_default_like=no_rake_default_like,
        use_occurrence_rate=use_occurrence_rate,
        parallel=parallel,
        return_one=return_one,
    )

    matched_rup_list = []
    for i, match in enumerate(match_results):
        if match is not None:
            matched_rup_list.append(match)

    matched_rups = into_outputs(matched_rup_list)

    return matched_rups


def into_outputs(matched_rup_list):
    """
    Modified and placed into new function by CB to get rup matching results
    into ideal format for use in subsequent steps of GEESE workflow.    
    """
    # If rups returned...
    if matched_rup_list != []:
        matched_rups = pd.concat(matched_rup_list, axis=0)
        if 'index' in matched_rups.columns:
            matched_rups = matched_rups.drop(columns=['index'])
        matched_rups = matched_rups.sort_values('likelihood', ascending=False)
        matched_rups = matched_rups.drop_duplicates(subset = ['likelihood'])
        matched_rups = matched_rups.drop_duplicates(subset = ['rups_for_event'])
        matched_rups["rake_diff"] = np.degrees(np.float_(matched_rups["rake_diff"]))
        matched_rups["attitude_diff"] = np.degrees(np.float_(
            matched_rups["attitude_diff"]))
    else:
        matched_rups = None
        
    return matched_rups