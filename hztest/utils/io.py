import os
import io
from typing import Optional, Union

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

from openquake.commonlib.logictree import SourceModelLogicTree
from openquake.hazardlib.source import (AreaSource, ComplexFaultSource,
                                        CharacteristicFaultSource,
                                        NonParametricSeismicSource,
                                        PointSource, MultiPointSource,
                                        SimpleFaultSource)

from .bins import SpacemagBin
from .model import read, _get_source_model


def sort_sources(brd):
    sorted_sources = {}

    for k, v in brd.items():
        sorted_sources[k] = {
            'area': [],
            'simple_fault': [],
            'complex_fault': [],
            'point': [],
            'nonpar': [],
            'multipoint': [],
            'none': []
        }
        for i, vv in enumerate(v):
            try:
                vs = read(vv,
                          get_info=False,
                          area_source_discretization=15.,
                          rupture_mesh_spacing=2.,
                          complex_fault_mesh_spacing=5.)
            except Exception as e:
                print('error reading ', k, vv, e)

            try:
                vs
            except NameError:
                break

            for j, source in enumerate(vs):
                if isinstance(source, AreaSource):
                    sorted_sources[k]['area'].append(source)
                elif isinstance(
                        source,
                    (SimpleFaultSource, CharacteristicFaultSource)):
                    sorted_sources[k]['simple_fault'].append(source)
                elif isinstance(source, ComplexFaultSource):
                    sorted_sources[k]['complex_fault'].append(source)
                elif isinstance(source, PointSource):
                    sorted_sources[k]['point'].append(source)
                elif isinstance(source, MultiPointSource):
                    sorted_sources[k]['multipoint'].append(source)
                elif isinstance(source, NonParametricSeismicSource):
                    sorted_sources[k]['nonpar'].append(source)
                elif isinstance(source, list):
                    for s in source:
                        if isinstance(s, AreaSource):
                            sorted_sources[k]['area'].append(s)
                        elif isinstance(
                                s,
                            (SimpleFaultSource, CharacteristicFaultSource)):
                            sorted_sources[k]['simple_fault'].append(s)
                        elif isinstance(s, ComplexFaultSource):
                            sorted_sources[k]['complex_fault'].append(s)
                        elif isinstance(s, PointSource):
                            sorted_sources[k]['point'].append(s)
                        elif isinstance(s, MultiPointSource):
                            sorted_sources[k]['multipoint'].append(s)
                        elif isinstance(s, NonParametricSeismicSource):
                            sorted_sources[k]['nonpar'].append(s)
                elif source is None:
                    #print(k, i, j, source)
                    sorted_sources[k]['none'].append(source)
                else:
                    print(type(source))

    return sorted_sources


def read_branch_sources(base_dir, lt_file='ssmLT.xml'):
    #base_dir ='../../../hazard_models/mosaic/{}/in/'.format(acr)

    lt = SourceModelLogicTree(os.path.join(base_dir, lt_file), validate=False)

    d = {}
    for k, v in lt.branches.items():
        try:
            d[k] = [base_dir + val for val in v.value.split()]
        except:
            print('error in ', k)
            pass

    return d


def process_source_logic_tree(base_dir: str,
                              lt_file: str = 'ssmLT.xml',
                              verbose: bool = False):
    if verbose:
        print('reading source branches')
    lt = sort_sources(read_branch_sources(base_dir, lt_file=lt_file))

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

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, yscale='log')
    plt.title('Magnitude-Frequency Distribution')

    if model is True:
        mod_mfd = sbin.get_rupture_mfd(cumulative=True)
        ax.plot(list(mod_mfd.keys()),
                list(mod_mfd.values()),
                model_format,
                label=model_label)

    if observed is True:
        obs_mfd = sbin.get_empirical_mfd(cumulative=True, t_yrs=observed_time)
        ax.plot(list(obs_mfd.keys()),
                list(obs_mfd.values()),
                observed_format,
                label=observed_label)

    ax.legend(loc='upper right')
    ax.set_ylabel('Annual frequency of exceedance')
    ax.set_xlabel('Magnitude')

    if save_fig is not False:
        fig.savefig(save_fig)

    if return_fig is True:
        return fig

    elif return_string is True:
        fig_str = io.StringIO()
        fig.savefig(fig_str, format='svg')
        plt.close(fig)
        #fig_svg = fig_str.getvalue()
        fig_svg = '<svg' + fig_str.getvalue().split('<svg')[1]
        return fig_svg


def write_mfd_plots_to_gdf(bin_gdf: GeoDataFrame, **kwargs):
    plot_series = bin_gdf['SpacemagBin'].apply(make_mfd_plot, **kwargs)
    bin_gdf['mfd_plots'] = plot_series