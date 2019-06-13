import os

from openquake.commonlib.logictree import SourceModelLogicTree
from openquake.hazardlib.source import (AreaSource, ComplexFaultSource,
                                        CharacteristicFaultSource,
                                        NonParametricSeismicSource,
                                        PointSource, MultiPointSource,
                                        SimpleFaultSource)

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
