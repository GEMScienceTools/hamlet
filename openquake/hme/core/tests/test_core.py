import os

from openquake.hme.core.core import read_yaml_config, get_test_lists_from_config

BASE_PATH = os.path.dirname(__file__)
# UNIT_TEST_DATA_DIR = os.path.join(BASE_PATH, "data", "unit_test_data")

yml_2 = os.path.join(BASE_PATH, "test_core_1.yml")

cfg_2 = read_yaml_config(yml_2, validate=False)


def test_read_yaml_config():
    cfg = read_yaml_config(yml_2, validate=False)
    assert cfg == {
        "input": {
            "bins": {"h3_res": 3, "mfd_bin_max": 9.0},
            "ssm": {
                "branch": "b1",
                "tectonic_region_types": ["Active Shallow Crust"],
                "source_types": None,
                "max_depth": None,
                "job_ini_file": None,
                "ssm_dir": "../../../../data/source_models/sm1/",
                "ssm_lt_file": "ssmLT.xml",
            },
            "rupture_file": {
                "rupture_file_path": None,
                "read_rupture_file": False,
                "save_rupture_file": False,
            },
            "subset": {"file": None, "buffer": 0.0},
            "simple_ruptures": True,
        },
        "meta": {"description": "Fake yaml for testing"},
        "config": {
            "model_framework": {
                "gem": {"likelihood": {"p1": 1, "p2": 2}},
                "sanity": {"max_check": {"warn": True}},
            }
        },
    }


def test_get_test_lists_from_config():
    test_dict = get_test_lists_from_config(cfg_2)

    assert test_dict["gem"] == ["likelihood"]
    assert test_dict["sanity"] == ["max_check"]
