import json
import logging
from typing import Union, Optional, Sequence

import h3
import pandas as pd
from tqdm import tqdm
from geopandas import GeoDataFrame
from shapely.geometry import Point
from openquake.hazardlib.geo.point import Point as oqPoint

from openquake.hazardlib.source.rupture import (
    float5,
    ParametricProbabilisticRupture,
    NonParametricProbabilisticRupture,
)


from ..utils import rupture_list_to_gdf
from ..simple_rupture import SimpleRupture, rup_to_dict
from openquake.hme.utils.io.source_processing import (
    _get_h3_cell_for_rupture_df,
)


def write_ruptures_to_file(
    rupture_gdf: GeoDataFrame,
    rupture_file_path: str,
    # simple_ruptures: bool = True,
):
    ruptures_out = rupture_gdf
    rup_file_type = rupture_file_path.split(".")[-1]
    if rup_file_type == "hdf5":
        ruptures_out.to_hdf(rupture_file_path, key="ruptures")
    elif rup_file_type == "feather":
        ruptures_out.reset_index().to_feather(rupture_file_path)
    elif rup_file_type == "csv":
        ruptures_out.to_csv(rupture_file_path, index=True)
    else:
        raise ValueError("Cannot write to {} filetype".format(rup_file_type))


def write_simple_ruptures_to_file(
    rupture_gdf: GeoDataFrame, rupture_file_path: str
):
    ruptures_out = rupture_gdf.drop("cell_id", axis=1)

    rup_file_type = rupture_file_path.split(".")[-1]
    if rup_file_type == "hdf5":
        ruptures_out.to_hdf(rupture_file_path, key="ruptures")
    elif rup_file_type == "feather":
        ruptures_out.to_feather(rupture_file_path)
    elif rup_file_type == "csv":
        ruptures_out.to_csv(rupture_file_path, index=True)
    else:
        raise ValueError("Cannot write to {} filetype".format(rup_file_type))


def write_simple_ruptures_to_file_old(
    rupture_gdf: GeoDataFrame, rupture_file_path: str
):
    ruptures_out = pd.DataFrame.from_dict(
        [rup_to_dict(rup) for rup in rupture_gdf["rupture"]]
    )

    rup_file_type = rupture_file_path.split(".")[-1]
    if rup_file_type == "hdf5":
        ruptures_out.to_hdf(rupture_file_path, key="ruptures")
    elif rup_file_type == "feather":
        ruptures_out.to_feather(rupture_file_path)
    elif rup_file_type == "csv":
        ruptures_out.to_csv(rupture_file_path, index=False)
    else:
        raise ValueError("Cannot write to {} filetype".format(rup_file_type))


def write_oq_ruptures_to_file(
    rupture_gdf: GeoDataFrame, rupture_file_path: str
):
    outfile_type = rupture_file_path.split(".")[-1]
    if outfile_type != "json":
        logging.warn("Writing JSON to {}".format(rupture_file_path))

    out_json = {
        "ruptures": [oq_rupture_to_json(rup) for rup in rupture_gdf["rupture"]]
    }

    with open(rupture_file_path, "w") as of:
        json.dump(out_json, of)


def oq_rupture_to_json(
    rupture: Union[
        ParametricProbabilisticRupture, NonParametricProbabilisticRupture
    ],
):
    mesh = surface_to_array(rupture.surface)

    rec = {}
    rec["id"] = rupture.rup_id
    rec["mag"] = rupture.mag
    rec["rake"] = rupture.rake
    rec["lon"] = rupture.hypocenter.x
    rec["lat"] = rupture.hypocenter.y
    rec["dep"] = rupture.hypocenter.z
    rec["trt"] = rupture.tectonic_region_type
    # rec['multiplicity'] = rup.multiplicity
    rec["mesh"] = json.dumps(
        [[[float5(z) for z in y] for y in x] for x in mesh]
    )

    return rec


def read_rupture_file(
    rupture_file, h3_res: int = 3, parallel=False
) -> pd.DataFrame:
    # parallel not currently used but leaving for future possibilities
    rup_file_type = rupture_file.split(".")[-1]

    if rup_file_type == "hdf5":
        rupture_df = pd.read_hdf(rupture_file, key="ruptures")
    elif rup_file_type == "feather":
        rupture_df = pd.read_feather(rupture_file)
    elif rup_file_type == "csv":
        rupture_df = pd.read_csv(rupture_file, index_col=0)
    else:
        raise ValueError("Cannot read filetype {}".format(rup_file_type))

    _get_h3_cell_for_rupture_df(rupture_df, h3_res)

    return rupture_df


def read_rupture_file_old(rupture_file):
    rup_file_type = rupture_file.split(".")[-1]

    if rup_file_type == "hdf5":
        ruptures = pd.read_hdf(rupture_file, key="ruptures")
    elif rup_file_type == "feather":
        ruptures = pd.read_feather(rupture_file)
    elif rup_file_type == "csv":
        ruptures = pd.read_csv(rupture_file)
    else:
        raise ValueError("Cannot read filetype {}".format(rup_file_type))

    logging.info("converting to SimpleRuptures")
    rupture_gdf = read_ruptures_from_dataframe(ruptures)

    return rupture_gdf


def _rupture_from_namedtuple(row):
    rup = SimpleRupture(
        strike=row.strike,
        dip=row.dip,
        rake=row.rake,
        mag=row.mag,
        hypocenter=oqPoint(row.lon, row.lat, row.depth),
        occurrence_rate=row.occurrence_rate,
        source=str(row.source),
    )
    return rup


def _process_ruptures_from_df(rup_df: pd.DataFrame):
    rup_list = list(
        tqdm(
            map(
                _rupture_from_namedtuple,
                rup_df.itertuples(index=False, name="rup"),
            ),
            total=len(rup_df),
        )
    )
    rupture_df = rupture_list_to_gdf(rup_list)
    return rupture_df


gem_flatfile_eq_cols = [
    "event_id",
    "event_time",
    "ISC_ev_id",
    "ev_latitude",
    "ev_longitude",
    "ev_depth_km",
    "fm_type_code",
    "ML",
    # "ML_ref",  # these were only in some source datasets and not used
    "Mw",
    # "Mw_ref",
    "Ms",
    # "Ms_ref",
    # "event_source_id",
    "es_strike",
    "es_dip",
    "es_rake",
    "es_z_top",
    "es_length",
    "es_width",
]


def read_ruptures_from_dataframe(rup_df):
    new_rup_df = _process_ruptures_from_df(rup_df)
    return new_rup_df


def load_flatfile(
    filepath,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
    h3_res: Optional[int] = None,
):
    """
    Loads a flatfile from a filepath, and processes it based on its
    filetype.

    Returns:
    eq_gm_df: a DataFrame of the earthquakes recorded
    gm_df: a DataFrame of the ground motions
    """
    file_extension = filepath.split(".")[-1]
    if file_extension == "csv":
        flatfile = read_flatfile_df(filepath)
        eq_gm_df, gm_df = process_flatfile_df(
            flatfile, min_mag=min_mag, max_mag=max_mag, h3_res=h3_res
        )

    # not sure if I should return an sql/gpkg database with both eq and gm
    # tables
    elif file_extension == "gpkg":
        flatfile = read_flatfile_gpkg(filepath)

    return eq_gm_df, gm_df


def read_flatfile_gpkg(filepath):
    raise NotImplementedError


def read_flatfile_df(filepath):
    # assuming GEM Global Flatfile format
    # flatfile = pd.read_csv(filepath, index_col=0) # older flatfile version
    flatfile = pd.read_csv(filepath)
    return flatfile


def process_flatfile(flatfile):
    if isinstance(flatfile, pd.DataFrame):
        return process_flatfile_df(flatfile)


def process_flatfile_df(
    flatfile: pd.DataFrame,
    eq_cols=gem_flatfile_eq_cols,
    index_col="event_id",
    h3_res: Optional[int] = None,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
):

    logging.info("Processing flatfile")

    if min_mag:
        flatfile = flatfile[flatfile.Mw >= min_mag]
    if max_mag:
        flatfile = flatfile[flatfile.Mw <= max_mag]

    eq_gm_df = flatfile[eq_cols].drop_duplicates(subset=index_col)

    gm_df = flatfile  # .set_index([index_col], append=True)

    convert_flatfile_eq_cols(eq_gm_df)

    def parse_geometry(row, x="longitude", y="latitude", z="depth"):
        if z:
            return Point(row[x], row[y], row[z])
        else:
            return Point(row[x], row[y])

    eq_gm_df["geometry"] = eq_gm_df.apply(parse_geometry, axis=1)
    gm_df["geometry"] = gm_df.apply(
        parse_geometry, axis=1, x="st_longitude", y="st_latitude", z=None
    )

    if h3_res is not None:
        eq_gm_df["cell_id"] = [
            h3.latlng_to_cell(row.latitude, row.longitude, h3_res)
            for i, row in eq_gm_df.iterrows()
        ]

        gm_df["cell_id"] = [
            h3.latlng_to_cell(row.st_latitude, row.st_longitude, h3_res)
            for i, row in gm_df.iterrows()
        ]

    return eq_gm_df, gm_df


def convert_flatfile_eq_cols(eq_gm_df: pd.DataFrame):
    conversion_dict = {
        #'event_id' same
        "ev_longitude": "longitude",
        "ev_latitude": "latitude",
        "ev_depth_km": "depth",
        "es_strike": "strike",
        "es_dip": "dip",
        "es_rake": "rake",
        "Mw": "magnitude",
        "event_time": "time",
    }

    eq_gm_df.rename(columns=conversion_dict, inplace=True)
