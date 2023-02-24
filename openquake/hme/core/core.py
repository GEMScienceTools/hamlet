"""
Core functions for running Hamlet.

The functions here read the configuration file, then load all of the model
inputs (seismic sources, observed earthquake catalog, etc.), run the tests, and
write the output.
"""

import os
import time
import logging
from copy import deepcopy
from typing import Union, Optional, Tuple
import pdb

import json
import yaml
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils.io import (
    # process_source_logic_tree,
    process_source_logic_tree_oq,
    # write_mfd_plots_to_gdf,
)

from openquake.hme.utils.validate_inputs import validate_cfg

from openquake.hme.utils import (
    deep_update,
    # rupture_dict_from_logic_tree_dict,
    # rupture_list_to_gdf,
    # rupture_dict_to_gdf,
    # rup_to_dict,
    # rup_df_from_dict,
    # read_ruptures_from_dataframe,
    # add_ruptures_to_bins,
    # add_earthquakes_to_bins,
    make_earthquake_gdf_from_csv,
    trim_eq_catalog,
    # make_bin_gdf_from_rupture_gdf,
    # subset_source,
    trim_inputs,
)

from openquake.hme.utils.results_processing import process_results

from openquake.hme.reporting import generate_basic_report


from openquake.hme.utils.io.source_processing import (
    rupture_dict_from_logic_tree_dict,
    rupture_dict_to_gdf,
)

from openquake.hme.utils.io import (
    # write_bin_gdf_to_csv,
    read_rupture_file,
    write_ruptures_to_file,
)
from openquake.hme.model_test_frameworks.gem.gem_tests import gem_test_dict
from openquake.hme.model_test_frameworks.relm.relm_tests import relm_test_dict
from openquake.hme.model_test_frameworks.sanity.sanity_checks import (
    sanity_test_dict,
)
from openquake.hme.model_test_frameworks.model_description import (
    model_description_test_dict,
)

Openable = Union[str, bytes, int, "os.PathLike[Any]"]

test_dict = {
    "gem": gem_test_dict,
    "relm": relm_test_dict,
    "sanity": sanity_test_dict,
    "model_description": model_description_test_dict,
}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

cfg_defaults = {
    "input": {
        "bins": {"h3_res": 3},
        "ssm": {
            "branch": None,
            "tectonic_region_types": None,
            "source_types": None,
            "max_depth": None,
            "job_ini_file": None,
            "ssm_dir": None,
            "ssm_lt_file": None,
        },
        "rupture_file": {
            "rupture_file_path": None,
            "read_rupture_file": False,
            "save_rupture_file": False,
        },
        "subset": {"file": None, "buffer": 0.0},
        "simple_ruptures": True,
    },
}


def read_yaml_config(
    yaml_config: Openable, fill_fields: bool = True, validate: bool = True
) -> dict:
    """
    Reads a model test configuration file (YAML).

    :param yaml_config:
        path or file-like object in the YAML format.

    :returns:
        Model test configuration from the YAML made into a dictionary.
    """
    cfg = deepcopy(cfg_defaults)

    logger.info("reading YAML configuration")
    with open(yaml_config) as config_file:
        cfg = deep_update(cfg, yaml.safe_load(config_file))

    if validate:
        validate_cfg(cfg)

    return cfg


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
        if hasattr(cfg["config"]["model_framework"][fw], "keys"):
            fw_test_names = list(cfg["config"]["model_framework"][fw].keys())
        else:
            fw_test_names = cfg["config"]["model_framework"][fw]

        tests[fw] = fw_test_names

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

    eq_gdf = make_earthquake_gdf_from_csv(
        seis_cat_file, **seis_cat_params, h3_res=cfg["input"]["bins"]["h3_res"]
    )

    if any(
        [d in seis_cat_cfg for d in ["stop_date", "start_date", "duration"]]
    ):
        start_date = seis_cat_cfg.get("start_date")
        stop_date = seis_cat_cfg.get("stop_date")
        duration = seis_cat_cfg.get("duration")
        eq_gdf = trim_eq_catalog(
            eq_gdf,
            start_date=start_date,
            stop_date=stop_date,
            duration=duration,
        )

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

    eq_gdf = make_earthquake_gdf_from_csv(
        pro_cat_file, **seis_cat_params, h3_res=cfg["input"]["bins"]["h3_res"]
    )

    if any(["stop_date", "start_date", "duration"]) in pro_cat_cfg:
        start_date = pro_cat_cfg.get("start_date")
        stop_date = pro_cat_cfg.get("stop_date")
        duration = pro_cat_cfg.get("duration")
        eq_gdf = trim_eq_catalog(
            eq_gdf,
            start_date=start_date,
            stop_date=stop_date,
            duration=duration,
        )

    return eq_gdf


def load_ruptures_from_file(cfg: dict):
    """
    Reads a flat file with ruptures.
    """
    h3_res = cfg["input"]["bins"]["h3_res"]
    parallel = cfg["config"]["parallel"]
    rup_file = cfg["input"]["rupture_file"]["rupture_file_path"]
    logging.info("Reading ruptures from {}".format(rup_file))
    if os.path.exists(rup_file):
        rupture_gdf = read_rupture_file(
            rup_file, h3_res=h3_res, parallel=parallel
        )

    else:
        logging.warn("Rupture file does not exist; reading SSM.")
        rupture_gdf = load_ruptures_from_ssm(cfg)

    return rupture_gdf


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
    ssm_lt_sources, weights, source_rup_counts = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        source_cfg["ssm_dir"],
        lt_file=source_cfg["ssm_lt_file"],
        source_types=source_cfg["source_types"],
        tectonic_region_types=source_cfg["tectonic_region_types"],
        branch=source_cfg["branch"],
        description=cfg["meta"]["description"],
    )

    logger.info("  making dictionary of ruptures")
    rupture_dict = rupture_dict_from_logic_tree_dict(
        ssm_lt_sources,
        source_rup_counts=source_rup_counts,
        parallel=cfg["config"]["parallel"],
    )

    del ssm_lt_sources

    logger.info("  making geodataframe from ruptures")
    rupture_gdf = rupture_dict_to_gdf(
        rupture_dict,
        weights,
    )
    logger.info("  done preparing rupture dataframe")

    return rupture_gdf


def load_inputs(cfg: dict) -> dict:
    """
    Loads all of the inputs specified by the `cfg` and returns a tuple of
    :class:`GeoDataFrame` objects, the earthquake catalog and the bins.

    :param cfg:
        Configuration for the evaluations, such as that parsed from the YAML
        config file.
    """

    eq_gdf = load_obs_eq_catalog(cfg)

    if cfg["input"]["rupture_file"]["read_rupture_file"] is True:
        rupture_gdf = load_ruptures_from_file(cfg)
    else:
        rupture_gdf = load_ruptures_from_ssm(cfg)

    if cfg["input"]["rupture_file"]["save_rupture_file"] is True:
        logging.info("Writing ruptures to file")
        write_ruptures_to_file(
            rupture_gdf,
            cfg["input"]["rupture_file"]["rupture_file_path"],
        )

    logging.info("grouping ruptures by cell")
    cell_groups = rupture_gdf.groupby("cell_id")

    logger.info("rupture_gdf shape: {}".format(rupture_gdf.shape))
    logger.debug(
        "rupture_gdf memory: {} GB".format(
            sum(rupture_gdf.memory_usage(index=True, deep=True)) * 1e-9
        )
    )

    if cfg["input"]["subset"]["file"] is not None:
        # logger.info("   Subsetting bin_gdf")
        # bin_gdf = subset_source(
        #    bin_gdf,
        #    subset_file=cfg["input"]["subset"]["file"],
        #    buffer=cfg["input"]["subset"]["buffer"],
        # )
        logger.warn("CANNOT SUBSET SOURCE YET!!!")

    logging.info("trimming earthquake catalog")
    cells_in_model = rupture_gdf.cell_id.unique()
    eq_in_model = (cell_id in cells_in_model for cell_id in eq_gdf.cell_id)
    eq_gdf = eq_gdf.loc[eq_in_model]

    logging.info("grouping earthquakes by cell")
    eq_groups = eq_gdf.groupby("cell_id")

    input_data = {
        "rupture_gdf": rupture_gdf,
        "cell_groups": cell_groups,
        "eq_gdf": eq_gdf,
        "eq_groups": eq_groups,
    }

    if "prospective_catalog" in cfg["input"].keys():
        logger.info("adding prospective earthquakes to input data")
        pro_gdf = load_pro_eq_catalog(cfg)
        pro_eq_in_model = (
            cell_id in cells_in_model for cell_id in pro_gdf.cell_id
        )
        pro_eq_gdf = pro_gdf.loc[pro_eq_in_model]
        input_data["pro_gdf"] = pro_eq_gdf
        input_data["pro_groups"] = pro_eq_gdf.groupby("cell_id")

    return input_data


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
    except KeyError:
        pass

    input_data = load_inputs(cfg)

    t_done_load = time.time()
    logger.info(
        "Done loading and preparing model in {0:.2f} s".format(
            t_done_load - t_start
        )
    )

    test_lists = get_test_lists_from_config(cfg)

    results = {}

    if "model_description" in test_lists.keys():
        mod_desc_tests = test_lists.pop("model_description")
        results["model_description"] = {
            test: {"val": test_dict["model_description"][test](cfg, input_data)}
            for test in mod_desc_tests
        }

    logger.info("trimming rupture and earthquake data to test magnitude range")
    trim_inputs(input_data, cfg)
    logger.info(" {:_} ruptures".format(len(input_data["rupture_gdf"])))

    for framework, tests in test_lists.items():
        results[framework] = {}
        for test in tests:
            results[framework][test] = {
                "val": test_dict[framework][test](cfg, input_data)
            }

    t_done_eval = time.time()
    logger.info(
        "Done evaluating model in {0:.2f} s".format(t_done_eval - t_done_load)
    )

    process_results(cfg, input_data, results)

    if "report" in cfg.keys():
        write_reports(cfg, results=results, input_data=input_data)

    if "json" in cfg.keys():
        # raise NotImplementedError()
        logging.warn("JSON output not implemented")
        write_json(cfg, results)

    t_out_done = time.time()
    logger.info(
        "Done writing outputs in {0:.2f} s".format(t_out_done - t_done_eval)
    )
    logger.info(
        "Done with everything in {0:.2f} m".format(
            (t_out_done - t_start) / 60.0
        )
    )

    return results


"""
output processing
"""


def write_json(cfg: dict, results: dict):
    out_results = {}

    for test_framework, test_results in results.items():
        if test_framework not in out_results.keys():
            out_results[test_framework] = {}
        for test, res in test_results.items():
            if test != "model_mfd":
                out_results[test_framework][test] = res["val"]

    with open(cfg["json"]["outfile"], "w") as f:
        json.dump(out_results, f)


def write_outputs(
    cfg: dict,
    bin_gdf: GeoDataFrame,
    eq_gdf: GeoDataFrame,
    write_index: bool = False,
) -> None:
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
        # write_mfd_plots_to_gdf(bin_gdf, **cfg["output"]["plots"]["kwargs"])
        raise NotImplementedError("can't do plots rn")

    if "map_epsg" in cfg["config"]:
        out_gdf = out_gdf.to_crs(cfg["config"]["map_epsg"])

    if "bin_gdf" in cfg["output"].keys():
        outfile = cfg["output"]["bin_gdf"]["file"]
        out_format = outfile.split(".")[-1]
        bin_gdf["bin_index"] = bin_gdf.index
        bin_gdf.index = np.arange(len(bin_gdf))

        if out_format == "csv":
            # write_bin_gdf_to_csv(outfile, bin_gdf)
            raise NotImplementedError("can't do plots rn")

        else:
            try:
                bin_gdf.drop("SpacemagBin", axis=1).to_file(
                    outfile,
                    driver=OUTPUT_FILE_MAP[out_format],
                    index=write_index,
                )
            except KeyError:
                raise Exception(f"No writer for {out_format} format")


OUTPUT_FILE_MAP = {"geojson": "GeoJSON"}


def write_reports(cfg: dict, results: dict, input_data: dict) -> None:
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
        generate_basic_report(cfg, results, input_data)
