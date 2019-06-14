import os
import time
import logging
from typing import Union

import yaml
import numpy as np
from geopandas import GeoDataFrame

from hztest.utils.io import process_source_logic_tree, write_mfd_plots_to_gdf
from hztest.utils import (make_SpacemagBins_from_bin_gis_file,
                          rupture_dict_from_logic_tree_dict,
                          rupture_list_to_gdf, add_ruptures_to_bins,
                          add_earthquakes_to_bins,
                          make_earthquake_gdf_from_csv)

from hztest.model_test_frameworks.gem.gem_tests import gem_test_dict
from hztest.model_test_frameworks.relm.relm_tests import relm_test_dict

Openable = Union[str, bytes, int, 'os.PathLike[Any]']

test_dict = {'model_framework': {'gem': gem_test_dict, 'relm': relm_test_dict}}
"""
config parsing
"""


def read_yaml_config(yaml_config: Openable) -> dict:
    """
    Reads a model test configuration file (YAML).

    :param yaml_config:
        path or file-like object in the YAML format.

    :returns:
        Model test configuration from the YAML made into a dictionary.
    """
    logging.info('reading YAML configuration')
    with open(yaml_config) as config_file:
        config = yaml.safe_load(config_file)
    return config


def update_defaults(cfg: dict):
    # need to update hztest defaults with values from the cfg,
    # or just add the necessary stuff to cfg
    raise NotImplementedError


def get_test_list_from_config(cfg: dict) -> list:
    # some ordering would be nice

    logging.info('getting tests from config')

    test_names = list(cfg['config']['tests'].keys())

    tds = test_dict['model_framework'][cfg['config']['model_framework']]

    tests = [tds[test] for test in test_names]

    return tests


"""
input processing
"""


def load_obs_eq_catalog(cfg: dict) -> GeoDataFrame:

    logging.info('making earthquake GDF from seismic catalog')

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
    Makes a GeoDataFrame of :class:`~hztest.utils.bins.SpacemagBin`s by passing
    the required parameters from the configuration dictionary to the
    :func:`~hztest.utils.make_SpacemagBins_from_bin_gis_file` function.

    :param cfg:
        Configuration for the test, such as that parsed from the YAML
        config file when running model tests.

    :returns:
        A GeoDataFrame of the SpacemagBins.
    """

    logging.info('making bin GDF from GIS file')

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

    logging.info('loading ruptuers into geodataframe')

    source_cfg: dict = cfg['input']['ssm']
    # make/fetch bin df?  Right now, no.

    ssm_lt_ruptures = process_source_logic_tree(
        source_cfg['ssm_dir'], lt_file=source_cfg['ssm_lt_file'])

    rupture_dict = rupture_dict_from_logic_tree_dict(
        ssm_lt_ruptures,
        source_types=source_cfg['source_types'],
        parallel=cfg['config']['parallel'])

    rupture_gdf = rupture_list_to_gdf(rupture_dict[source_cfg['branch']])
    return rupture_gdf


def load_inputs(cfg: dict):

    # TODO: figure out whether to load EQs based on which tests to run

    bin_gdf = make_bin_gdf(cfg)
    rupture_gdf = load_ruptures_from_ssm(cfg)

    logging.info('adding ruptures to bins')
    add_ruptures_to_bins(rupture_gdf, bin_gdf)

    eq_gdf = load_obs_eq_catalog(cfg)

    logging.info('adding earthquakes to bins')
    add_earthquakes_to_bins(eq_gdf, bin_gdf)

    return bin_gdf, eq_gdf


"""
running tests
"""


def run_tests(cfg):

    try:
        np.random.seed(cfg['config']['rand_seed'])
    except Exception as e:
        logging.warning('Cannot use random seed: {}'.format(e.__str__()))

    tests = get_test_list_from_config(cfg)

    bin_gdf, eq_gdf = load_inputs(cfg)

    for test in tests:
        test(cfg, bin_gdf=bin_gdf, obs_seis_catalog=eq_gdf)

    write_outputs(cfg, bin_gdf=bin_gdf, eq_gdf=eq_gdf)


"""
output processing
"""


def write_outputs(cfg, bin_gdf: GeoDataFrame, eq_gdf: GeoDataFrame):

    logging.info('writing outputs')

    if 'plots' in cfg['output'].keys():
        write_mfd_plots_to_gdf(bin_gdf, **cfg['output']['plots']['kwargs'])

    if 'bin_gdf' in cfg['output'].keys():
        bin_gdf.drop('SpacemagBin',
                     axis=1).to_file(cfg['output']['bin_gdf']['file'],
                                     driver='GeoJSON')
