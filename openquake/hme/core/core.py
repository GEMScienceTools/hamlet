import time
import logging
from typing import Union, Optional

import yaml
import numpy as np
from geopandas import GeoDataFrame

from openquake.hme.utils.io import (process_source_logic_tree,
                                    write_mfd_plots_to_gdf)
from openquake.hme.utils import (make_SpacemagBins_from_bin_gis_file,
                                 rupture_dict_from_logic_tree_dict,
                                 rupture_list_to_gdf, add_ruptures_to_bins,
                                 add_earthquakes_to_bins,
                                 make_earthquake_gdf_from_csv,
                                 make_bin_gdf_from_rupture_gdf)
from openquake.hme.utils.reporting import generate_basic_report

from openquake.hme.model_test_frameworks.gem.gem_tests import gem_test_dict
from openquake.hme.model_test_frameworks.relm.relm_tests import relm_test_dict

Openable = Union[str, bytes, int, 'os.PathLike[Any]']

test_dict = {'model_framework': {'gem': gem_test_dict, 'relm': relm_test_dict}}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def read_yaml_config(yaml_config: Openable, fill_fields: bool = True) -> dict:
    """
    Reads a model test configuration file (YAML).

    :param yaml_config:
        path or file-like object in the YAML format.

    :returns:
        Model test configuration from the YAML made into a dictionary.
    """
    logger.info('reading YAML configuration')
    with open(yaml_config) as config_file:
        cfg = yaml.safe_load(config_file)

    if fill_fields:
        _fill_necessary_fields(cfg)

    return cfg


def update_defaults(cfg: dict):
    # need to update openquake.hme defaults with values from the cfg,
    # or just add the necessary stuff to cfg
    raise NotImplementedError


def _fill_necessary_fields(cfg: dict):
    """
    Fills the configuration dictionary with `None` types for optional
    parameters that were not included in the YAML file.

    """
    # to fill in as necessary (oh god that comment)

    necessary_fields = {
        'input': {
            'ssm': ['branch', 'tectonic_region_types', 'source_types']
        }
    }

    for field, subfield in necessary_fields.items():
        for sub_name, subsubfields in subfield.items():
            for subsubname in subsubfields:
                if subsubname not in cfg[field][sub_name].keys():
                    logger.warning(
                        f"['{field}']['{sub_name}']['{subsubname}'] filled")
                    cfg[field][sub_name][subsubname] = None


def get_test_list_from_config(cfg: dict) -> list:
    # some ordering would be nice

    logger.info('getting tests from config')

    test_names = list(cfg['config']['tests'].keys())

    tds = test_dict['model_framework'][cfg['config']['model_framework']]

    tests = [tds[test] for test in test_names]

    return tests


"""
input processing
"""


def load_obs_eq_catalog(cfg: dict) -> GeoDataFrame:

    logger.info('making earthquake GDF from seismic catalog')

    seis_cat_cfg: dict = cfg['input']['seis_catalog']
    seis_cat_params = {
        k: v
        for k, v in seis_cat_cfg['columns'].items() if v is not None
    }
    seis_cat_file = seis_cat_cfg['seis_catalog_file']

    eq_gdf = make_earthquake_gdf_from_csv(seis_cat_file, **seis_cat_params)

    return eq_gdf


def make_bin_gdf(cfg: dict) -> GeoDataFrame:
    """
    Makes a GeoDataFrame of :class:`~openquake.hme.utils.bins.SpacemagBin`s by
    passing the required parameters from the configuration dictionary to the
    :func:`~openquake.hme.utils.make_SpacemagBins_from_bin_gis_file` function.

    :param cfg: Configuration for the test, such as that parsed from the YAML
        config file when running model tests.

    :returns: A GeoDataFrame of the SpacemagBins.
    """

    logger.info('making bin GDF from GIS file')

    bin_cfg: dict = cfg['input']['bins']

    bin_gdf = make_SpacemagBins_from_bin_gis_file(
        bin_cfg['bin_gis_file'],
        min_mag=bin_cfg['mfd_bin_min'],
        max_mag=bin_cfg['mfd_bin_max'],
        bin_width=bin_cfg['mfd_bin_width'])

    return bin_gdf


def load_ruptures_from_ssm(cfg: dict):
    """
    Reads a seismic source model, processes it, and returns a GeoDataFrame with
    the ruptures.  All necessary information is passed from the `cfg`
    dictionary, as from a test configuration file.

    :param cfg:
        Configuration for the test, such as that parsed from the YAML
        config file when running model tests.

    :returns:
        A GeoDataFrame of the ruptures.
    """

    logger.info('loading ruptures into geodataframe')

    source_cfg: dict = cfg['input']['ssm']

    logger.info('  processing logic tree')
    ssm_lt_ruptures = process_source_logic_tree(
        source_cfg['ssm_dir'],
        lt_file=source_cfg['ssm_lt_file'],
        source_types=source_cfg['source_types'],
        tectonic_region_types=source_cfg['tectonic_region_types'],
        branch=source_cfg['branch'])

    logger.info('  making dictionary of ruptures')
    rupture_dict = rupture_dict_from_logic_tree_dict(
        ssm_lt_ruptures, parallel=cfg['config']['parallel'])

    del ssm_lt_ruptures

    logger.info('  making geodataframe from ruptures')
    rupture_gdf = rupture_list_to_gdf(rupture_dict[source_cfg['branch']])
    logger.info('  done preparing rupture dataframe')

    return rupture_gdf


def load_inputs(cfg: dict):

    # TODO: figure out whether to load EQs based on which tests to run

    #bin_gdf = make_bin_gdf(cfg)
    rupture_gdf = load_ruptures_from_ssm(cfg)
    bin_gdf = make_bin_gdf_from_rupture_gdf(
        rupture_gdf,
        res=3,
        min_mag=cfg['input']['bins']['mfd_bin_min'],
        max_mag=cfg['input']['bins']['mfd_bin_max'],
        bin_width=cfg['input']['bins']['mfd_bin_width'],
    )

    logger.info('bin_gdf shape: {}'.format(bin_gdf.shape))

    logger.info('rupture_gdf shape: {}'.format(rupture_gdf.shape))
    logger.debug('rupture_gdf memory: {} GB'.format(
        sum(rupture_gdf.memory_usage(index=True, deep=True)) * 1e-9))

    logger.info('adding ruptures to bins')
    add_ruptures_to_bins(rupture_gdf,
                         bin_gdf,
                         parallel=cfg['config']['parallel'])

    logger.debug('bin_gdf memory: {} GB'.format(
        sum(bin_gdf.memory_usage(index=True, deep=True)) * 1e-9))

    eq_gdf = load_obs_eq_catalog(cfg)

    logger.info('adding earthquakes to bins')
    add_earthquakes_to_bins(eq_gdf, bin_gdf)

    return bin_gdf, eq_gdf


"""
running tests
"""


def run_tests(cfg: dict):

    t_start = time.time()

    try:
        np.random.seed(cfg['config']['rand_seed'])
    except Exception as e:
        logger.warning('Cannot use random seed: {}'.format(e.__str__()))

    tests = get_test_list_from_config(cfg)

    bin_gdf, eq_gdf = load_inputs(cfg)

    t_done_load = time.time()
    logger.info(
        'Done loading and preparing model in {0:.2f} s'.format(t_done_load -
                                                               t_start))

    results = {}
    # make dict w/ test fn as key, name as val to fill results while testing
    tds = test_dict['model_framework'][cfg['config']['model_framework']]
    test_inv = {fn: name for name, fn in tds.items() if fn in tests}

    for test in tests:
        results[test_inv[test]] = {
            'val': test(cfg, bin_gdf=bin_gdf, obs_seis_catalog=eq_gdf)
        }

    t_done_eval = time.time()
    logger.info('Done evaluating model in {0:.2f} s'.format(t_done_eval -
                                                            t_done_load))

    if 'output' in cfg.keys():
        write_outputs(cfg, bin_gdf=bin_gdf, eq_gdf=eq_gdf)

    if 'report' in cfg.keys():
        write_reports(cfg, bin_gdf=bin_gdf, eq_gdf=eq_gdf, results=results)

    t_out_done = time.time()
    logger.info('Done writing outputs in {0:.2f} s'.format(t_out_done -
                                                           t_done_eval))
    logger.info('Done with everything in {0:.2f} m'.format(
        (t_out_done - t_start) / 60.))


"""
output processing
"""


def write_outputs(cfg: dict, bin_gdf: GeoDataFrame, eq_gdf: GeoDataFrame):

    logger.info('writing outputs')

    if 'plots' in cfg['output'].keys():
        write_mfd_plots_to_gdf(bin_gdf, **cfg['output']['plots']['kwargs'])

    if 'bin_gdf' in cfg['output'].keys():
        bin_gdf['bin_index'] = bin_gdf.index
        bin_gdf.drop('SpacemagBin', axis=1).to_file(
            cfg['output']['bin_gdf']['file'],
            driver='GeoJSON',
        )


def write_reports(cfg: dict,
                  results: dict,
                  bin_gdf: Optional[GeoDataFrame] = None,
                  eq_gdf: Optional[GeoDataFrame] = None):
    logger.info('writing reports')

    if 'basic' in cfg['report'].keys():
        generate_basic_report(cfg, results, bin_gdf=bin_gdf, eq_gdf=eq_gdf)
