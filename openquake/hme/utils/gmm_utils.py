import logging

import numpy as np

from openquake.hazardlib.calc.gmf import ground_motion_fields
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.geodetic import point_at
from openquake.hazardlib.site import Site, SiteCollection

from openquake.hazardlib.imt import PGA, PGV

from openquake.hazardlib.source.rupture import (
    BaseRupture,
    EBRupture,
)

from openquake.hazardlib.geo.surface import PlanarSurface

from openquake.smt.comparison.utils_gmpes import _get_z1, _get_z25

from openquake.hme.utils.stats import geom_mean

from openquake.hme.utils.utils import breakpoint


def build_oq_rupture(rupture):
    if hasattr(rupture, "surface"):
        surface = rupture.surface
    else:
        # maybe w/ hz.source.rupture.build_planar
        raise NotImplementedError
    oqrup = BaseRupture(
        mag=rupture.magnitude,
        rake=rupture.rake,
        surface=surface,
        hypocenter=surface.get_middle_point(),
        tectonic_region_type=rupture.tectonic_region_type,
    )
    oqrup.ztor = surface.get_top_edge_depth()
    return oqrup


def make_sitecol(
    lons,
    lats,
    vs30s=650.0,
    vs30s_meas_type=None,
) -> SiteCollection:
    sites = []

    def get_param(p, i):
        if p is not None:
            if np.isscalar(p):
                return p
            else:
                return p[i]
        else:
            return p

    for i, lon in enumerate(lons):
        site_args = {"location": Point(lon, lats[i])}
        site_args["vs30"] = get_param(vs30s, i)
        if get_param(vs30s_meas_type, i) == "measured":
            site_args["vs30measured"] = 1
        if get_param(vs30s_meas_type, i) == "inferred":
            site_args["vs30measured"] = 0
        site_args["z1pt0"] = _get_z1(site_args["vs30"], "global")
        # site_args["z2pt5s"] = get_param(z2pt5s, i)

        sites.append(Site(**site_args))

    return SiteCollection(sites)


def gmf_from_rupture(
    rupture,
    sites=None,
    imts=[
        PGA(),
    ],
    gsim=None,
    truncation_level=3,
    realizations=1,
    correlation_model=None,
    seed=420,
    return_dists=True,
):

    # should probably do something here

    return ground_motion_fields(
        rupture,
        sites=sites,
        imts=imts,
        gsim=gsim,
        truncation_level=truncation_level,
        realizations=realizations,
        correlation_model=correlation_model,
        seed=seed,
    )


def get_imls_from_flatfile_row(row, imts):
    imts_ = []
    for imt in imts:
        try:
            imts_.append(imt.__name__)
        except AttributeError:
            imts_.append(imt.__repr__())

    imt_funcs = {"PGA": get_pga_from_flatfile_row}

    imt_results = {imt: imt_funcs[imt](row) for imt in imts_}

    return imt_results


def get_pga_from_flatfile_row(row):
    # from chris:
    # first try to get geom mean of 2-component horizontal pga
    # if not this, then try to get rotd50 and convert?
    # pga and SA are converted to cm/s^2
    if (not np.isnan(row.U_pga)) and (not np.isnan(row.V_pga)):
        pga_cm_s2 = geom_mean(abs(row.U_pga), abs(row.V_pga))
    else:
        if not np.isnan(row.rotD50_pga):
            pga_cm_s2 = row.rotD50_pga
        else:
            logging.warning(
                f"can't find horizontal PGA values for eq {row.name}"
            )
            pga_cm_s2 = np.nan

    pga_g = pga_cm_s2 * 0.01 / 9.81
    return pga_g


def get_proper_distance(rupture: BaseRupture, sites, distance_key):
    if distance_key == "rjb":
        pass
    elif distance_key == "rrup":
        pass
    elif distance_key == "rx":
        pass


def make_rup_from_flatfile(eq, trt=None, default_trt="Active Shallow Crust"):

    # todo: get standard msr from trt, make rup from_hypocenter if needed

    strike = eq.strike
    dip = eq.dip
    ztor = eq.es_z_top
    rake = eq.rake
    width = eq.es_width
    length = eq.es_length
    lon = eq.longitude
    lat = eq.latitude
    mag = eq.magnitude

    if trt is None:
        trt = eq.event_trt_from_classifier  # may be null
        if trt is None:
            trt = default_trt

    height = width * np.sin(np.radians(dip))
    hdist = width * np.cos(np.radians(dip))

    if ztor is not None:
        depth = ztor + height / 2

    # Move hor. 1/2 hdist in direction -90
    mid_top = point_at(lon, lat, strike - 90, hdist / 2)
    # Move hor. 1/2 hdist in direction +90
    mid_bot = point_at(lon, lat, strike + 90, hdist / 2)

    # compute corner points at the surface
    top_right = point_at(mid_top[0], mid_top[1], strike, length / 2)
    top_left = point_at(mid_top[0], mid_top[1], strike + 180, length / 2)
    bot_right = point_at(mid_bot[0], mid_bot[1], strike, length / 2)
    bot_left = point_at(mid_bot[0], mid_bot[1], strike + 180, length / 2)

    # compute corner points in 3D; rounded to 5 digits to avoid having
    # slightly different surfaces between macos and linux
    pbl = Point(bot_left[0], bot_left[1], depth + height / 2).round()
    pbr = Point(bot_right[0], bot_right[1], depth + height / 2).round()
    ptl = Point(top_left[0], top_left[1], depth - height / 2).round()
    ptr = Point(top_right[0], top_right[1], depth - height / 2).round()

    surface = PlanarSurface.from_corner_points(ptl, ptr, pbr, pbl)

    rup = BaseRupture(mag, rake, trt, Point(lon, lat, depth), surface)

    rup.event_id = eq.event_id

    return rup
