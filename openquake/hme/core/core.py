"""
Core functions for running Hamlet.

The functions here read the configuration file, then load all of the model
inputs (seismic sources, observed earthquake catalog, etc.), run the tests, and
write the output.
"""

import time
import logging
from typing import Union, Optional, Tuple

import yaml
import numpy as np
from geopandas import GeoDataFrame

from openquake.hme.utils.io import process_source_logic_tree, write_mfd_plots_to_gdf
from openquake.hme.utils import (
    make_SpacemagBins_from_bin_gis_file,
    rupture_dict_from_logic_tree_dict,
    rupture_list_to_gdf,
    add_ruptures_to_bins,
    add_earthquakes_to_bins,
    make_earthquake_gdf_from_csv,
    make_bin_gdf_from_rupture_gdf,
)
from openquake.hme.reporting import generate_basic_report

from openquake.hme.model_test_frameworks.gem.gem_tests import gem_test_dict
from openquake.hme.model_test_frameworks.relm.relm_tests import relm_test_dict
from openquake.hme.model_test_frameworks.sanity.sanity_checks import sanity_test_dict

Openable = Union[str, bytes, int, "os.PathLike[Any]"]

test_dict = {"gem": gem_test_dict, "relm": relm_test_dict, "sanity": sanity_test_dict}

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
    logger.info("reading YAML configuration")
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
        "input": {"ssm": ["branch", "tectonic_region_types", "source_types"]}
    }

    for field, subfield in necessary_fields.items():
        for sub_name, subsubfields in subfield.items():
            for subsubname in subsubfields:
                if subsubname not in cfg[field][sub_name].keys():
                    logger.warning(f"['{field}']['{sub_name}']['{subsubname}'] filled")
                    cfg[field][sub_name][subsubname] = None


def get_test_lists_from_config(cfg: dict) -> dict:
    """
    Reads through the `cfg` and makes a dict of lists of tests or evaluations to
    run for each framework.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.
    """
    tests = {}
    frameworks = list(cfg["config"]["model_framework"].keys())

    for fw in frameworks:
        fw_test_names = list(cfg["config"]["model_framework"][fw].keys())
        tests[fw] = [test_dict[fw][test] for test in fw_test_names]

    return tests


"""
input processing
"""


def load_obs_eq_catalog(cfg: dict) -> GeoDataFrame:
    """
    Loads the observed earthquake catalog into a `GeoDataFrame` that has all of
    the earthquakes processed into :class:`~openquake.hme.utils.Earthquake`
    objects.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :returns:
        :class:`GeoDataFrame`
    """

    logger.info("making earthquake GDF from seismic catalog")

    seis_cat_cfg: dict = cfg["input"]["seis_catalog"]
    seis_cat_params = {
        k: v for k, v in seis_cat_cfg["columns"].items() if v is not None
    }
    seis_cat_file = seis_cat_cfg["seis_catalog_file"]

    eq_gdf = make_earthquake_gdf_from_csv(seis_cat_file, **seis_cat_params)

    return eq_gdf


def load_pro_eq_catalog(cfg: dict) -> GeoDataFrame:
    """
    Loads the prospective earthquake catalog into a `GeoDataFrame` that has all of
    the earthquakes processed into :class:`~openquake.hme.utils.Earthquake`
    objects. The formatting must be identical to the 'observed' seismic catalog.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :returns:
        :class:`GeoDataFrame`
    """

    logger.info("making earthquake GDF from seismic catalog")

    seis_cat_cfg: dict = cfg["input"]["seis_catalog"]
    pro_cat_cfg: dict = cfg["input"]["prospective_catalog"]
    seis_cat_params = {
        k: v for k, v in seis_cat_cfg["columns"].items() if v is not None
    }
    pro_cat_file = pro_cat_cfg["prospective_catalog_file"]

    eq_gdf = make_earthquake_gdf_from_csv(pro_cat_file, **seis_cat_params)

    return eq_gdf


def make_bin_gdf(cfg: dict) -> GeoDataFrame:
    """
    Makes a GeoDataFrame of :class:`~openquake.hme.utils.bins.SpacemagBin`s by
    passing the required parameters from the configuration dictionary to the
    :func:`~openquake.hme.utils.make_SpacemagBins_from_bin_gis_file` function.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :returns: A GeoDataFrame of the SpacemagBins.
    """

    logger.info("making bin GDF from GIS file")

    bin_cfg: dict = cfg["input"]["bins"]

    bin_gdf = make_SpacemagBins_from_bin_gis_file(
        bin_cfg["bin_gis_file"],
        min_mag=bin_cfg["mfd_bin_min"],
        max_mag=bin_cfg["mfd_bin_max"],
        bin_width=bin_cfg["mfd_bin_width"],
    )

    return bin_gdf


def load_ruptures_from_ssm(cfg: dict):
    """
    Reads a seismic source model, processes it, and returns a GeoDataFrame with
    the ruptures.  All necessary information is passed from the `cfg`
    dictionary, as from a test configuration file.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :returns:
        A GeoDataFrame of the ruptures.
    """

    logger.info("loading ruptures into geodataframe")

    source_cfg: dict = cfg["input"]["ssm"]

    logger.info("  processing logic tree")
    ssm_lt_ruptures = process_source_logic_tree(
        source_cfg["ssm_dir"],
        lt_file=source_cfg["ssm_lt_file"],
        source_types=source_cfg["source_types"],
        tectonic_region_types=source_cfg["tectonic_region_types"],
        branch=source_cfg["branch"],
    )

    logger.info("  making dictionary of ruptures")
    rupture_dict = rupture_dict_from_logic_tree_dict(
        ssm_lt_ruptures, parallel=cfg["config"]["parallel"]
    )

    del ssm_lt_ruptures

    logger.info("  making geodataframe from ruptures")
    rupture_gdf = rupture_list_to_gdf(rupture_dict[source_cfg["branch"]])
    logger.info("  done preparing rupture dataframe")

    return rupture_gdf


def load_inputs(cfg: dict) -> Tuple[GeoDataFrame]:
    """
    Loads all of the inputs specified by the `cfg` and returns a tuple of
    :class:`GeoDataFrame` objects, the earthquake catalog and the bins.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.
    """
    rupture_gdf = load_ruptures_from_ssm(cfg)
    bin_gdf = make_bin_gdf_from_rupture_gdf(
        rupture_gdf,
        res=3,
        min_mag=cfg["input"]["bins"]["mfd_bin_min"],
        max_mag=cfg["input"]["bins"]["mfd_bin_max"],
        bin_width=cfg["input"]["bins"]["mfd_bin_width"],
    )

    logger.info("bin_gdf shape: {}".format(bin_gdf.shape))

    logger.info("rupture_gdf shape: {}".format(rupture_gdf.shape))
    logger.debug(
        "rupture_gdf memory: {} GB".format(
            sum(rupture_gdf.memory_usage(index=True, deep=True)) * 1e-9
        )
    )

    logger.info("adding ruptures to bins")
    add_ruptures_to_bins(rupture_gdf, bin_gdf)

    del rupture_gdf

    logger.debug(
        "bin_gdf memory: {} GB".format(
            sum(bin_gdf.memory_usage(index=True, deep=True)) * 1e-9
        )
    )

    eq_gdf = load_obs_eq_catalog(cfg)

    logger.info("adding earthquakes to bins")
    add_earthquakes_to_bins(eq_gdf, bin_gdf)

    if "prospective_catalog" in cfg["input"].keys():
        logger.info("adding prospective earthquakes to bins")
        pro_gdf = load_pro_eq_catalog(cfg)
        add_earthquakes_to_bins(pro_gdf, bin_gdf, category="prospective")
        return bin_gdf, eq_gdf, pro_gdf

    else:
        return bin_gdf, eq_gdf


"""
running tests
"""


def run_tests(cfg: dict) -> None:
    """
    Main Hamlet function.

    This function reads the `cfg`, loads all of the inputs, runs the
    evaluations, and then writes the ouputs.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    """

    t_start = time.time()

    try:
        np.random.seed(cfg["config"]["rand_seed"])
    except Exception as e:
        logger.warning("Cannot use random seed: {}".format(e.__str__()))

    if "prospective_catalog" in cfg["input"].keys():
        bin_gdf, eq_gdf, pro_gdf = load_inputs(cfg)
    else:
        bin_gdf, eq_gdf = load_inputs(cfg)
        pro_gdf = None

    t_done_load = time.time()
    logger.info(
        "Done loading and preparing model in {0:.2f} s".format(t_done_load - t_start)
    )

    test_lists = get_test_lists_from_config(cfg)
    test_inv = {
        framework: {
            fn: name for name, fn in test_dict[framework].items() if fn in fw_tests
        }
        for framework, fw_tests in test_lists.items()
    }

    results = {}

    for framework, tests in test_lists.items():
        results[framework] = {}
        for test in tests:
            results[framework][test_inv[framework][test]] = {
                "val": test(cfg, bin_gdf=bin_gdf)
            }

    t_done_eval = time.time()
    logger.info("Done evaluating model in {0:.2f} s".format(t_done_eval - t_done_load))

    if "output" in cfg.keys():
        write_outputs(cfg, bin_gdf=bin_gdf, eq_gdf=eq_gdf)

    if "report" in cfg.keys():
        write_reports(cfg, bin_gdf=bin_gdf, eq_gdf=eq_gdf, results=results)

    t_out_done = time.time()
    logger.info("Done writing outputs in {0:.2f} s".format(t_out_done - t_done_eval))
    logger.info(
        "Done with everything in {0:.2f} m".format((t_out_done - t_start) / 60.0)
    )


"""
output processing
"""


def write_outputs(cfg: dict, bin_gdf: GeoDataFrame, eq_gdf: GeoDataFrame) -> None:
    """
    Writes output GIS files and plots (i.e., maps or MFD plots.)

    All of the options for what to write are specified in the `cfg`.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :param bin_gdf:
        :class:`GeoDataFrame` with the spatial bins for testing

    :param eq_gdf:
        :class:`GeoDataFrame` with the observed earthquake catalog.
    """

    logger.info("writing outputs")

    if "plots" in cfg["output"].keys():
        write_mfd_plots_to_gdf(bin_gdf, **cfg["output"]["plots"]["kwargs"])

    if "bin_gdf" in cfg["output"].keys():
        bin_gdf["bin_index"] = bin_gdf.index
        bin_gdf.drop("SpacemagBin", axis=1).to_file(
            cfg["output"]["bin_gdf"]["file"], driver="GeoJSON",
        )


def write_reports(
    cfg: dict,
    results: dict,
    bin_gdf: Optional[GeoDataFrame] = None,
    eq_gdf: Optional[GeoDataFrame] = None,
) -> None:
    """
    Writes reports summarizing the results of the tests and evaluations.

    All of the options for what to write are specified in the `cfg`.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.

    :param results:
        Dictionary of results for the tests in each framework used.

    :param bin_gdf:
        :class:`GeoDataFrame` with the spatial bins for testing

    :param eq_gdf:
        :class:`GeoDataFrame` with the observed earthquake catalog.
    """
    logger.info("writing reports")

    if "basic" in cfg["report"].keys():
        generate_basic_report(cfg, results, bin_gdf=bin_gdf, eq_gdf=eq_gdf)
