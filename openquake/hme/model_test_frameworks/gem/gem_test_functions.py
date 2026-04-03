"""
Utility functions for running tests in the GEM model test framework.
"""

import os
import logging
from multiprocessing import Pool

import h3
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from openquake.hazardlib import imt as imt_module
from openquake.hazardlib import scalerel, valid
from openquake.hazardlib.contexts import ContextMaker, RuptureContext
from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.source.rupture import BaseRupture

from openquake.smt.residuals.gmpe_residuals import Residuals
from openquake.smt.residuals.context_db import ContextDB
from openquake.smt.residuals.residual_plotter import (
    ResidualPlot,
    ResidualWithMagnitude,
    ResidualWithDistance,
    ResidualWithVs30,
)

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


def model_mfd_eval_fn(
    rup_gdf,
    eq_gdf,
    mag_bins,
    t_yrs=None,
    completeness_table=None,
    annualize=False,
    stop_date=None,
):

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
    dists = get_distances(eq, rupture_df)
    return rupture_df.iloc[dists.argmin()]


class HamletContextDB(ContextDB):
    """
    Custom ContextDB that builds contexts from hamlet's earthquake
    DataFrame and GEM Global Flatfile records, suitable for SMT
    residual analysis.
    """
    def __init__(self, eq_df, gm_df, trt):
        """
        :param eq_df:
            DataFrame of unique earthquakes.
        :param gm_df:
            DataFrame of the GEM Global Flatfile.
        :param trt:
            Tectonic region type string (e.g. "Active Shallow Crust") of the
            given event.
        """
        self.eq_df = eq_df
        self.gm_df = gm_df
        self.trt = trt
        self.msr, self.aratio = self._get_rup_props_for_trt(trt)

    def _get_rup_props_for_trt(self, trt):
        """
        Return an appropriate MSR and aspect ratio for the TRT.
        """
        trt_lower = trt.lower()
        if "slab" in trt_lower or "intraslab" in trt_lower:
            return scalerel.strasser2010.StrasserIntraslab(), 5.0
        elif "interface" in trt_lower:
            return scalerel.strasser2010.StrasserInterface(), 5.0
        return scalerel.WC1994(), 2.0

    def get_contexts(self, imts):
        """
        Build contexts directly from hamlet DataFrames.
        """
        ctxs = []
        for idx, eq in self.eq_df.iterrows():
            
            # Get records
            records = self.gm_df[self.gm_df["event_id"] == eq.event_id]
            
            if len(records) == 0:
                continue
            
            ctx = RuptureContext()
            n_sites = len(records)

            # Get rup, site and distance params
            self._set_rupture_params(ctx, eq, self.msr)
            self._set_site_params(ctx, records, n_sites)
            self._set_distance_params(ctx, records, n_sites)

            ctx.sids = np.arange(n_sites, dtype=np.uint32)

            # Build an SMT-style ctx
            dic = {"EventID": eq.event_id, "Ctx": ctx}
            dic["Observations"] = {}
            dic["Retained"] = {}
            for imtx in imts:
                col_name = self._imt_to_rotd50_col(imtx)
                values = records[col_name].values.astype(float)
                check = pd.notnull(values)
                dic["Observations"][imtx] = np.asarray(
                    values, dtype=float
                )
                dic["Retained"][imtx] = np.argwhere(check).flatten()
            dic["Num. Sites"] = n_sites

            ctxs.append(dic)

        return ctxs

    def _set_rupture_params(self, ctx, eq):
        """
        Set rupture parameters on the context from the earthquake row.
        """
        # Mag and hypocenter
        ctx.mag = float(eq.magnitude)
        ctx.hypo_lon = float(eq.longitude)
        ctx.hypo_lat = float(eq.latitude)
        ctx.hypo_depth = float(eq.depth)

        # Nodal plane
        ctx.strike = (float(eq.strike) if pd.notnull(eq.get("strike")) else 0.0)
        ctx.dip = (float(eq.dip) if pd.notnull(eq.get("dip")) else 90.0)
        ctx.rake = (float(eq.rake) if pd.notnull(eq.get("rake")) else 0.0)

        # Set a ztor if we have one
        if pd.notnull(eq.get("es_z_top")):
            ctx.ztor = float(eq.es_z_top)
        else:
            ctx.ztor = float(eq.depth)

        # Try set the rupture width or compute a proxy if not available
        if pd.notnull(eq.get("es_width")):
            ctx.width = float(eq.es_width)
        else:
            ctx.width = np.sqrt(
                scalerel.WC1994().get_median_area(ctx.mag, ctx.rake))

        # Arbitrary hypocentral location (it's the relative distance
        # to the sites that matters)
        ctx.hypo_loc = (0.5, 0.5)

    def _set_site_params(self, ctx, records, n_sites):
        """
        Set site parameters on the context from flatfile records.
        """
        # Basic site params
        ctx.lons = records["st_longitude"].values.astype(float)
        ctx.lats = records["st_latitude"].values.astype(float)
        depths = records["st_elevation"].values.astype(float) * -1.0e-3
        depths[np.isnan(depths)] = 0.0
        ctx.depths = depths
        ctx.vs30 = records["vs30_m_sec"].values.astype(float)
        z1 = records["z1pt0 (m)"].values.astype(float)
        ctx.z1pt0 = np.where(np.isnan(z1), np.nan, z1)
        z2 = records["z2pt5 (km)"].values.astype(float)
        ctx.z2pt5 = np.where(np.isnan(z2), np.nan, z2)
        ctx.vs30measured = (
            records["vs30_meas_type"].str.strip().str.lower().eq(
                "measured").values)
        ctx.backarc = records["st_backarc"].astype(bool).values

    def _set_distance_params(self, ctx, records, n_sites):
        """
        Set distance parameters on the context from flatfile records.
        """
        # Point-source based
        ctx.repi = records["epi_dist"].values.astype(float)
        ctx.rhypo = np.sqrt(ctx.repi ** 2 + ctx.hypo_depth ** 2)

        # Finite rupture based
        ctx.rjb = records["JB_dist"].values.astype(float)
        ctx.rrup = records["rup_dist"].values.astype(float)
        ctx.rx = records["Rx_dist"].values.astype(float)
        ctx.ry0 = records["Ry0_dist"].values.astype(float)

        # Not used currently in SMT but needed so set as zeroed out
        ctx.rvolc = np.zeros(n_sites)
        ctx.rcdpp = np.zeros(n_sites)

        self._fill_missing_distances(ctx)

    def _fill_missing_distances(self, ctx):
        """
        Fill NaN distances by reconstructing a finite rupture.
        """
        dist_attrs = ["rrup", "rjb", "rx", "ry0", "repi", "rhypo"]
        has_missing = any(
            np.any(np.isnan(getattr(ctx, attr)))
            for attr in dist_attrs
        )
        if not has_missing:
            return

        try: # I use a "try-except" because we might get a rupture that is too
             # large for given Mw and MSR without setting a ztor depth constraint
             # which is a bit tricky in a coarse-level residual analysis like this
            hypoc = Point(ctx.hypo_lon, ctx.hypo_lat, ctx.hypo_depth)
            srf = PlanarSurface.from_hypocenter(
                hypoc, self.msr, ctx.mag, self.aratio,
                ctx.strike, ctx.dip, ctx.rake, ctx.ztor
            )
            rup = BaseRupture(ctx.mag, ctx.rake, None, hypoc, srf)

            # Build a dummy GMM that requires all distance types
            gmpe = valid.gsim("DummyGMPE")
            orig_r = list(gmpe.REQUIRES_DISTANCES)
            for d in ["repi", "rrup", "rjb", "rhypo", "rx", "ry0", "rvolc"]:
                if d not in orig_r:
                    orig_r.append(d)
            gmpe.REQUIRES_DISTANCES = frozenset(orig_r)

            mag_str = [f"{ctx.mag:.2f}"]
            oqp = {"imtls": {"PGA": []}, "mags": mag_str} # Dummy imtls here
            ctxm = ContextMaker(
                self.trt, [gmpe], oqp
            )

            for i in range(len(ctx.lons)):
                needs_fill = any(
                    np.isnan(getattr(ctx, attr)[i])
                    for attr in dist_attrs
                )
                if not needs_fill:
                    continue

                site = SiteCollection(
                    [
                        Site(
                            Point(ctx.lons[i], ctx.lats[i], ctx.depths[i]),
                            ctx.vs30[i],
                            ctx.z1pt0[i]
                            if not np.isnan(ctx.z1pt0[i])
                            else None,
                            ctx.z2pt5[i]
                            if not np.isnan(ctx.z2pt5[i])
                            else None,
                        )
                    ]
                )

                site_ctxs = ctxm.get_ctxs([rup], site)
                site_ctx = site_ctxs[0]

                for attr in dist_attrs:
                    if np.isnan(getattr(ctx, attr)[i]):
                        getattr(ctx, attr)[i] = float(getattr(site_ctx, attr)[0])

        except Exception as e:
            logging.warning(f"Could not fill missing distances: {e}")

    def _imt_to_rotd50_col(self, imtx):
        """
        Map IMT string to the GEM flatfile rotD50 column name.
        """
        if imtx == "PGA":
            return "rotD50_pga"
        elif "SA(" in imtx:
            period = imt_module.from_string(imtx).period
            period_str = str(period).replace(".", "_")
            return f"rotD50_T{period_str}"
        else:
            raise ValueError(f"Unsupported IMT: {imtx}")


def _generate_residual_plots(residuals, imts, output_dir):
    """
    Generate residual plots for all GMMs and IMTs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gmpe in residuals.gmpe_list:
        gmpe_str = str(gmpe).replace(" ", "_")
        for imtx in imts:
            prefix = os.path.join(output_dir, f"{gmpe_str}_{imtx}")
            ResidualPlot(
                residuals, gmpe, imtx, f"{prefix}_hist.png"
            )
            ResidualWithMagnitude(
                residuals, gmpe, imtx, f"{prefix}_vs_mag.png"
            )
            ResidualWithDistance(
                residuals, gmpe, imtx, f"{prefix}_vs_dist.png",
                distance_type="rrup",
            )
            ResidualWithVs30(
                residuals, gmpe, imtx, f"{prefix}_vs_vs30.png"
            )


def _assign_trt_to_earthquakes(test_config, input_data):
    """
    Run rupture matching and assign a TRT to each earthquake.
    """
    match_results = rupture_matching_eval_fn(
        input_data["rupture_gdf"],
        input_data["eq_gm_df"],
        distance_lambda=test_config["distance_lambda"],
        mag_window=test_config["mag_window"],
        group_return_threshold=test_config["group_return_threshold"],
        no_attitude_default_like=test_config["no_attitude_default_like"],
        no_rake_default_like=test_config["no_rake_default_like"],
        use_occurrence_rate=test_config["use_occurrence_rate"],
        return_one=test_config["return_one"],
        parallel=test_config["parallel"],
    )

    eq_trt_map = {}
    match_rups = test_config.get("match_rups", False)

    if match_rups and len(match_results["matched_rups"]) > 0:
        match_results["matched_rups"]["event_id"] = (
            input_data["eq_gm_df"]
            .loc[match_results["matched_rups"].index]
            .event_id
        )
        for idx, matched_rup in match_results["matched_rups"].iterrows():
            eq_trt_map[idx] = matched_rup.tectonic_region_type

    if not match_rups:
        match_results["unmatched_eqs"] = input_data["eq_gm_df"]

    for idx, eq in match_results["unmatched_eqs"].iterrows():
        if idx not in eq_trt_map:
            closest = get_closest_rupture(eq, input_data["rupture_gdf"])
            eq_trt_map[idx] = closest.tectonic_region_type

    return eq_trt_map


def evaluate_gmc(test_config, input_data):
    """
    Evaluate the GMMs for each TRT in the SSC. Return some plots
    summarising the performance of each GMM in each TRT for some
    IMTs of general interest
    """
    # Hardcode the GMMs to the GRM IMTs for now
    imts = ["PGA", "SA(0.3)", "SA(0.6)", "SA(1.0)"]

    logging.info("Matching ruptures to GM Earthquakes")
    eq_trt_map = _assign_trt_to_earthquakes(test_config, input_data)

    # Make out dir
    output_dir = test_config.get("output_dir", "gm_residual_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Group events by TRT and compute residuals using the SMT
    results = {}

    for trt in set(eq_trt_map.values()):
        eq_indices = [idx for idx, t in eq_trt_map.items() if t == trt]
        eq_subset = input_data["eq_gm_df"].loc[eq_indices]

        if len(eq_subset) == 0:
            continue

        gmpe_list = list(input_data["gsim_lt"].values.get(trt, []))

        logging.info(
            f"Computing residuals for TRT: {trt} "
            f"({len(eq_subset)} events, {len(gmpe_list)} GMMs)"
        )

        ctx_db = HamletContextDB(eq_subset, input_data["gm_df"], trt)

        residuals = Residuals(gmpe_list, imts)
        residuals.compute_residuals(ctx_db, component="rotD50")

        trt_dir = os.path.join(output_dir, trt.replace(" ", "_"))
        _generate_residual_plots(residuals, imts, trt_dir)

        results[trt] = residuals

    return results
