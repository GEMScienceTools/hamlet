import os
import logging
from typing import Union, Optional, Sequence

import pandas as pd
from geopandas import GeoDataFrame

from openquake.commonlib.logictree import SourceModelLogicTree
from openquake.hazardlib.source import (AreaSource, ComplexFaultSource,
                                        CharacteristicFaultSource,
                                        NonParametricSeismicSource,
                                        PointSource, MultiPointSource,
                                        SimpleFaultSource)

from .bins import SpacemagBin
from .model import read
from .plots import plot_mfd
from .simple_rupture import SimpleRupture, rup_to_dict


def _source_to_series(source):

    if isinstance(source, AreaSource):
        source_type = 'area'
    elif isinstance(source, (SimpleFaultSource, CharacteristicFaultSource)):
        source_type = 'simple_fault'
    elif isinstance(source, ComplexFaultSource):
        source_type = 'complex_fault'
    elif isinstance(source, PointSource):
        source_type = 'point'
    elif isinstance(source, MultiPointSource):
        source_type = 'multipoint'
    elif isinstance(source, NonParametricSeismicSource):
        source_type = 'nonpar'
    else:
        return

    return pd.Series({
        'source': source,
        'tectonic_region_type': source.tectonic_region_type,
        'source_type': source_type
    })


def sort_sources(branch_sources: dict,
                 source_types: Optional[Sequence[str]] = None,
                 tectonic_region_types: Optional[Sequence[str]] = None,
                 branch: Optional[str] = None) -> dict:
    """
    Creates lists of sources for each branch of interest, optionally filtering
    sources by `source_type` and `tectonic_region_type`.

    :param source_types:
        Types of sources to collect sources from. Any values in 
        (`simple_fault1, `complex_fault`, `area`, `point`, `multipoint`) 
        are allowed.  Must be passed as a sequence (i.e., a tuple or list).
        Specify `None` if all values are to be included.

    :param tectonic_region_types:
        Types of tectonic regions to collect sources from. Any values allowed
        in OpenQuake are allowed here. Must be passed as a sequence (i.e., a
        tuple or list).  Specify `None` if all values are to be included.

    :param branch:
        Branch to be evaluated; other branches will be skipped if this is not
        `None`.

    :returns:
        Dictionary with one list of sources per branch considered.
    """

    branch_source_lists = {}

    for branch_name, source_file_list in branch_sources.items():
        if branch_name == branch or branch is None:

            source_list = []

            for source_file in source_file_list:
                try:
                    sources_from_file = read(source_file,
                                             get_info=False,
                                             area_source_discretization=15.,
                                             rupture_mesh_spacing=2.,
                                             complex_fault_mesh_spacing=5.)
                except Exception as e:
                    logging.warning(
                        f'error reading {branch} {source_file}: {e}')

                try:
                    sources_from_file
                except NameError:
                    break

                for source in sources_from_file:
                    if isinstance(source, list):
                        source_list.extend(
                            [_source_to_series(s) for s in source])
                    elif source is None:
                        pass
                    else:
                        source_list.append(_source_to_series(source))

            if len(source_list) == 1:
                source_df = source_list[0].to_frame().transpose()
            else:
                source_df = pd.concat(source_list, axis=1).transpose()

            logging.info(f'source df shape:{source_df.shape}')

            if source_types is not None:
                source_df = source_df[source_df.source_type.isin(source_types)]
            if tectonic_region_types is not None:
                source_df = source_df[source_df.tectonic_region_type.isin(
                    tectonic_region_types)]

            branch_source_lists[branch_name] = source_df['source'].to_list()

    return branch_source_lists


def read_branch_sources(base_dir,
                        lt_file='ssmLT.xml',
                        branch: Optional[str] = None):
    lt = SourceModelLogicTree(os.path.join(base_dir, lt_file))

    d = {}
    for branch_name, branch_filename in lt.branches.items():
        if branch_name == branch or branch is None:
            try:
                d[branch_name] = [
                    base_dir + val for val in branch_filename.value.split()
                ]
            except:
                print('error in ', branch_name)
                pass

    return d


def process_source_logic_tree(base_dir: str,
                              lt_file: str = 'ssmLT.xml',
                              branch: Optional[str] = None,
                              source_types: Optional[Sequence] = None,
                              tectonic_region_types: Optional[Sequence] = None,
                              verbose: bool = False):
    if verbose:
        print('reading source branches')
    branch_sources = (read_branch_sources(base_dir, lt_file=lt_file))
    lt = sort_sources(branch_sources,
                      source_types=source_types,
                      tectonic_region_types=tectonic_region_types,
                      branch=branch)

    if verbose:
        print(lt.keys())

    return lt


def write_ruptures_to_file(rupture_gdf: GeoDataFrame, rupture_file_path: str):
    ruptures_out = pd.DataFrame.from_dict(
        [rup_to_dict(rup) for rup in rupture_gdf["rupture"]])

    rup_file_type = rupture_file_path.split(".")[-1]
    if rup_file_type == "hdf5":
        ruptures_out.to_hdf(rupture_file_path, key="ruptures")
    elif rup_file_type == "feather":
        ruptures_out.to_feather(rupture_file_path)
    elif rup_file_type == "csv":
        ruptures_out.to_csv(rupture_file_path, index=False)
    else:
        raise ValueError("Cannot write to {} filetype".format(rup_file_type))


def read_rupture_file(rupture_file):
    rup_file_type = rupture_file.split(".")[-1]

    if rup_file_type == "hdf5":
        ruptures = pd.read_hdf(rupture_file, key="ruptures")
    elif rup_file_type == "feather":
        ruptures = pd.read_feather(rupture_file)

    logging.info("converting to SimpleRuptures")

    rupture_gdf = read_ruptures_from_dataframe(ruptures)

    return rupture_gdf


def _rupture_from_df_row(row):
    rup = SimpleRupture(
        strike=row["strike"],
        dip=row["dip"],
        rake=row["rake"],
        mag=row["mag"],
        hypocenter=OQPoint(row["lon"], row["lat"], row["depth"]),
        occurrence_rate=row["occurrence_rate"],
        source=row["source"],
    )
    return rup


def _rupture_from_namedtuple(row):
    rup = SimpleRupture(
        strike=row.strike,
        dip=row.dip,
        rake=row.rake,
        mag=row.mag,
        hypocenter=OQPoint(row.lon, row.lat, row.depth),
        occurrence_rate=row.occurrence_rate,
        source=row.source,
    )
    return rup


def _process_ruptures_from_df(rup_df: pd.DataFrame):
    rup_list = list(
        tqdm(
            map(_rupture_from_namedtuple,
                rup_df.itertuples(index=False, name="rup")),
            total=len(rup_df),
        ))
    rupture_df = rupture_list_to_gdf(rup_list)
    return rupture_df


def read_ruptures_from_dataframe(rup_df):
    new_rup_df = _process_ruptures_from_df(rup_df)
    return new_rup_df


def make_mfd_plot(sbin: SpacemagBin,
                  model: bool = True,
                  model_format: str = 'C0-',
                  model_label: str = 'model',
                  observed: bool = False,
                  observed_time: float = 1.,
                  observed_format: str = 'C1o-.',
                  observed_label: str = 'observed',
                  return_fig: bool = True,
                  return_string: bool = False,
                  save_fig: Union[bool, str] = False,
                  **kwargs):
    """
    :param save_fig:
        Either the filename to save to, or specify `False`.
    """
    if model is True:
        mod_mfd = sbin.get_rupture_mfd(cumulative=True)
    else:
        mod_mfd = None

    if observed is True:
        obs_mfd = sbin.get_empirical_mfd(cumulative=True, t_yrs=observed_time)
    else:
        obs_mfd = None

    return plot_mfd(model=mod_mfd,
                    model_format=model_format,
                    model_label=model_label,
                    observed=obs_mfd,
                    observed_format=observed_format,
                    observed_label=observed_label,
                    return_fig=return_fig,
                    return_string=return_string,
                    save_fig=save_fig,
                    **kwargs)


def write_mfd_plots_to_gdf(bin_gdf: GeoDataFrame, **kwargs):
    plot_series = bin_gdf['SpacemagBin'].apply(make_mfd_plot,
                                               model_iters=0,
                                               **kwargs)
    bin_gdf['mfd_plots'] = plot_series


def write_bin_gdf_to_csv(filename, bin_gdf: GeoDataFrame, index: bool = False):
    bin_gdf['wkt'] = bin_gdf.apply(lambda row: row.geometry.to_wkt(), axis=1)

    bin_wkt = bin_gdf.drop(['geometry', 'SpacemagBin'], axis=1)

    bin_wkt.to_csv(filename, index=index)
