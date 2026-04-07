"""
Utility functions for GMM residual analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
import re

from openquake.hazardlib import imt as imt_module
from openquake.hazardlib import scalerel, valid
from openquake.hazardlib.contexts import ContextMaker, RuptureContext
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
    plot_residual_means_and_stds_with_period,
)

from openquake.hme.model_test_frameworks.gem.gem_test_functions import (
    rupture_matching_eval_fn,
    get_closest_rupture,
)


class HamletContextDB(ContextDB):
    """
    Custom ContextDB that builds contexts from hamlet's earthquake
    DataFrame and GEM Global Flatfile records, suitable for SMT
    residual analysis.
    """
    def __init__(self, eq_df, gm_df, oq_rup):
        """
        :param eq_df:
            DataFrame of unique earthquakes.
        :param gm_df:
            DataFrame of the GEM Global Flatfile.
        :param oq_rup:
            Dict mapping eq index to matched model ruptures. The rupture
            parameters in the ctxs are taken from the matched ruptures.
        """
        self.eq_df = eq_df
        self.gm_df = gm_df
        self.oq_rup = oq_rup
        first_rup = next(iter(oq_rup.values())) # TRT is in the rup attributes
        self.trt = first_rup["tectonic_region_type"]
        self.msr, self.aratio = self._get_rup_props_for_trt(self.trt)

    def _get_rup_props_for_trt(self, trt):
        """
        Return an appropriate MSR and aspect ratio for the TRT.
        """
        trt_lower = trt.lower()
        if "slab" in trt_lower or "intraslab" in trt_lower:
            return scalerel.strasser2010.StrasserIntraslab(), 5.0
        elif "interface" in trt_lower:
            return scalerel.strasser2010.StrasserInterface(), 5.0
        else:
            return scalerel.WC1994(), 2.0

    def get_contexts(self, nodal_plane_index, imts, component):
        """
        Build contexts directly from hamlet DataFrame.
        """
        ctxs = []
        for idx, eq in self.eq_df.iterrows():

            # Get records
            records = self.gm_df[self.gm_df["event_id"] == eq.event_id]

            if len(records) == 0:
                continue

            ctx = RuptureContext()
            n_sites = len(records)

            # Use matched model rupture params
            self._set_rupture_params(ctx, self.oq_rup[idx])
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
                self.msr.get_median_area(ctx.mag, ctx.rake))

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
        ctx.z1pt0 = records["z1pt0 (m)"].values.astype(float)
        ctx.z2pt5 = records["z2pt5 (km)"].values.astype(float)
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
        Fill empty distances by reconstructing a finite rupture.
        """
        dist_attrs = ["rrup", "rjb", "rx", "ry0", "repi", "rhypo"]
        has_missing = any(
            np.any(np.isnan(getattr(ctx, attr)))
            for attr in dist_attrs
        )
        if not has_missing:
            return

        try: # Use a "try-except" because we might get a rupture that is too
             # large for given Mw, MSR and aratio combo without setting a ztor 
             # depth constraint which is a bit tricky in a coarse-level residual
             # analysis like this to always handle automatically
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
                            else None, # Will be set to -999 in the sm_database to turn off basin adjustment
                            ctx.z2pt5[i]
                            if not np.isnan(ctx.z2pt5[i])
                            else None, # Same again here
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
            int_part = int(period)
            frac_part = round((period - int_part) * 1000)
            return f"rotD50_T{int_part}_{frac_part:03d}"
        else:
            raise ValueError(f"Unsupported IMT: {imtx}")


def generate_residual_plots(residuals, imts, output_dir):
    """
    Generate residual plots for all GMMs and IMTs.
    Per-IMT plots are saved in per-GMM subdirectories.
    A summary plot of residual means and std devs vs period is also saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gmpe in residuals.gmpe_list:
        # Straight from GSIM XML so they are full toml representations
        gmpe_str = re.sub(
            r'[^\w\-.]', '_', str(gmpe)).split("___toml")[0]
        gmpe_dir = os.path.join(output_dir, gmpe_str)
        os.makedirs(gmpe_dir, exist_ok=True)
        for imtx in imts:
            prefix = os.path.join(gmpe_dir, f"{gmpe_str}_{imtx}")
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

    # Plot residual means and std devs vs period (all GMMs on one figure)
    plot_residual_means_and_stds_with_period(
        residuals,
        os.path.join(output_dir, "residual_means_stds_vs_period.png"),
    )


def rup_match_and_assign_trts(test_config, input_data):
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
    eq_rup_map = {}
    match_rups = test_config.get("match_rups", False)

    if match_rups and len(match_results["matched_rups"]) > 0:
        match_results["matched_rups"]["event_id"] = (
            input_data["eq_gm_df"]
            .loc[match_results["matched_rups"].index]
            .event_id
        )
        for idx, matched_rup in match_results["matched_rups"].iterrows():
            eq_trt_map[idx] = matched_rup["tectonic_region_type"]
            eq_rup_map[idx] = matched_rup

    if not match_rups:
        match_results["unmatched_eqs"] = input_data["eq_gm_df"]

    for idx, eq in match_results["unmatched_eqs"].iterrows():
        if idx not in eq_trt_map:
            closest = get_closest_rupture(eq, input_data["rupture_gdf"])
            eq_trt_map[idx] = closest["tectonic_region_type"]
            eq_rup_map[idx] = closest

    return eq_trt_map, eq_rup_map


def evaluate_gmc(test_config, input_data):
    """
    Evaluate the GMMs for each TRT in the SSC. Return some plots
    summarising the performance of each GMM in each TRT for some
    IMTs of general interest
    """
    # Hardcode the GMMs to the GRM IMTs for now
    candidate_imts = ["PGA", "SA(0.3)", "SA(0.6)", "SA(1.0)"]

    # Filter to IMTs that have data in the flatfile
    gm_df = input_data["gm_df"]

    logging.info("Matching ruptures to GM Earthquakes")
    eq_trt_map, eq_rup_map = rup_match_and_assign_trts(test_config, input_data)

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

        # Filter to records for this TRT's earthquakes
        trt_records = gm_df[gm_df["event_id"].isin(eq_subset["event_id"])]

        # Keep only earthquakes with more than 2 recordings
        rec_counts = trt_records.groupby("event_id").size()
        val_ids = rec_counts[rec_counts > 2].index
        trt_records = trt_records[trt_records["event_id"].isin(val_ids)]
        eq_subset = eq_subset[eq_subset["event_id"].isin(val_ids)]

        # Only proceed if enough data (above the min of 3 records)
        if len(eq_subset) == 0:
            logging.info(
                f"No events with 3 or more recordings for TRT {trt}, skipping"
            )
            continue

        # Check which IMTs have data in these records
        imts = []
        for imtx in candidate_imts:
            col = HamletContextDB._imt_to_rotd50_col(None, imtx)
            if trt_records[col].notna().any():
                imts.append(imtx)
            else:
                logging.info(
                    f"Skipping IMT {imtx}: no data for TRT {trt}"
                )

        if not imts:
            logging.warning(
                f"No IMTs with data for TRT {trt}, skipping"
            )
            continue

        gmpe_list = list(input_data["gsim_lt"].values.get(trt))

        logging.info(
            f"Computing residuals for TRT: {trt} "
            f"({len(eq_subset)} events, {len(gmpe_list)} GMMs, "
            f"{len(imts)} IMTs)"
        )

        trt_oq_rup = {i: eq_rup_map[i] for i in eq_indices}
        ctx_db = HamletContextDB(
            eq_subset, input_data["gm_df"], oq_rup=trt_oq_rup
        )

        residuals = Residuals(gmpe_list, imts)
        residuals.compute_residuals(ctx_db, component="rotD50")

        trt_dir = os.path.join(output_dir, trt.replace(" ", "_"))
        generate_residual_plots(residuals, imts, trt_dir)

        results[trt] = residuals

    return results
