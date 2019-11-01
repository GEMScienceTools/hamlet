import os
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

    branch_source_lists = {}

    for branch_name, source_file_list in branch_sources.items():
        if branch_name == branch or branch is None:

            source_list = []

            for i, source_file in enumerate(source_file_list):
                try:
                    sources_from_file = read(source_file,
                                             get_info=False,
                                             area_source_discretization=15.,
                                             rupture_mesh_spacing=2.,
                                             complex_fault_mesh_spacing=5.)
                except Exception as e:
                    print('error reading ', branch, source_file, e)

                try:
                    sources_from_file
                except NameError:
                    break

                for j, source in enumerate(sources_from_file):
                    if isinstance(source, list):
                        source_list.append(
                            *[_source_to_series(s) for s in source])
                    elif source is None:
                        pass
                    else:
                        source_list.append(_source_to_series(source))

            if len(source_list) == 1:
                source_df = source_list[0].to_frame().transpose()
            else:
                source_df = pd.concat(source_list)

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
    lt = SourceModelLogicTree(os.path.join(base_dir, lt_file), validate=False)

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
        Either the filename to save to, or 
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
    plot_series = bin_gdf['SpacemagBin'].apply(make_mfd_plot, **kwargs)
    bin_gdf['mfd_plots'] = plot_series
