import json
import logging
from typing import Union  # , Optional, Sequence

# from h3 import h3
import pandas as pd
from tqdm import tqdm
from geopandas import GeoDataFrame
from openquake.hazardlib.geo.point import Point

from openquake.hazardlib.source.rupture import (
    float5,
    ParametricProbabilisticRupture,
    NonParametricProbabilisticRupture,
)


from ..utils import rupture_list_to_gdf
from ..simple_rupture import SimpleRupture, rup_to_dict
from openquake.hme.utils.io.source_processing import (
    _get_h3_cell_for_rupture_df,
    _get_h3_cell_for_rupture_df_parallel,
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
        ruptures_out.to_feather(rupture_file_path)
    elif rup_file_type == "csv":
        ruptures_out.to_csv(rupture_file_path)
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
        ruptures_out.to_csv(rupture_file_path, index=False)
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
    ]
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
    rup_file_type = rupture_file.split(".")[-1]

    if rup_file_type == "hdf5":
        rupture_df = pd.read_hdf(rupture_file, key="ruptures")
    elif rup_file_type == "feather":
        rupture_df = pd.read_feather(rupture_file)
    elif rup_file_type == "csv":
        rupture_df = pd.read_csv(rupture_file, index_col=0)
    else:
        raise ValueError("Cannot read filetype {}".format(rup_file_type))

    if parallel is False:
        _get_h3_cell_for_rupture_df(rupture_df, h3_res)
    else:
        _get_h3_cell_for_rupture_df_parallel(rupture_df, h3_res)

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
        hypocenter=Point(row.lon, row.lat, row.depth),
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


def read_ruptures_from_dataframe(rup_df):
    new_rup_df = _process_ruptures_from_df(rup_df)
    return new_rup_df
