import os
import pathlib
import unittest

from openquake.hazardlib.source import SimpleFaultSource

from openquake.hme.core.core import read_yaml_config
from openquake.hme.utils.tests import load_sm1
from openquake.hme.utils.io.source_reader import (
    csm_from_job_ini,
    # get_csm_rlzs,
    process_source_logic_tree_oq,
    make_composite_source,
    # get_branch_weights,
    make_job_ini,
)


BASE_PATH = pathlib.Path(os.path.dirname(__file__))


source_cfg = load_sm1.cfg["input"]["ssm"]

# w/ job ini
# get job ini
# get csm
# process sources
# - one branch
# - all branches (may need second test model?)
#  - get weights
#  - make composie source


def test_single_branch_without_job_ini():
    def test_make_job_ini():
        job_ini = make_job_ini(
            source_cfg["ssm_dir"],
            lt_file=source_cfg["ssm_lt_file"],
            description=load_sm1.cfg["meta"]["description"],
        )

        # not sure how to easily deal with full paths during testing
        job_ini_result = {
            "calculation_mode": "preclassical",
            "description": "test",
            "rupture_mesh_spacing": "2.0",
            "area_source_discretization": "15.0",
            "width_of_mfd_bin": "0.1",
            "maximum_distance": "200",
            "investigation_time": "1.0",
            "source_model_logic_tree": "ssmLT.xml",
            "gsim_logic_tree": "gmmLT.xml",
            "reference_vs30_type": "measured",
            "reference_vs30_value": "800.0",
            "reference_depth_to_1pt0km_per_sec": "30.0",
            "truncation_level": "3.0",
            "job_ini": "<in-memory>",
            "ground_motion_fields": "False",
            "intensity_measure_types_and_levels": "{'PGA': [0.5]}",
            "inputs": {"source_model_logic_tree": "/ssmLT.xml"},
        }

        for k in job_ini.keys():
            if k == "source_model_logic_tree":
                ssm_file = job_ini[k].split("/")[-1]
                ssm_file_result = job_ini_result[k].split("/")[-1]
                assert ssm_file == ssm_file_result
            elif k == "gsim_logic_tree":
                gmm_file = job_ini[k].split("/")[-1]
                gmm_file_result = job_ini_result[k].split("/")[-1]
                assert gmm_file == gmm_file_result
            elif k == "inputs":
                assert (
                    job_ini[k]["source_model_logic_tree"].split("/")[-1]
                    == job_ini_result[k]["source_model_logic_tree"].split("/")[
                        -1
                    ]
                )

            else:
                assert job_ini[k] == job_ini_result[k]

        return job_ini

    job_ini = test_make_job_ini()

    def test_csm_from_job_ini():
        csm, sources, source_info, gmm_filepath = csm_from_job_ini(job_ini)

        assert csm.count_ruptures() == 7797
        # not really sure what can be easily checked, because
        # the csm is made of a bunch of OQ classes that I don't
        # want try to to store and load

        return csm, sources, source_info

    csm, _sources, _source_info = test_csm_from_job_ini()


def test_process_source_logic_tree_oq():
    (
        ssm_lt_sources,
        ssm_lt_weights,
        ssm_lt_rup_counts,
        gsim_lt,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        source_cfg["ssm_dir"],
        lt_file=source_cfg["ssm_lt_file"],
        source_types=source_cfg["source_types"],
        tectonic_region_types=source_cfg["tectonic_region_types"],
        branch=source_cfg["branch"],
        description=load_sm1.cfg["meta"]["description"],
    )

    assert list(ssm_lt_sources.keys()) == [0]
    assert len(ssm_lt_sources[0]) == 18
    assert isinstance(ssm_lt_sources[0][0], SimpleFaultSource)

    assert list(ssm_lt_weights.keys()) == [0]
    assert ssm_lt_weights == {0: 1.0}
    

def test_2_branches_compound():
    test_dir = (BASE_PATH / '..' / '..' / 'tests' / 'data' / 'source_models' / 
                '2_branches')
    cfg = read_yaml_config(test_dir / 'test_2_ssm_branches.yaml')
    source_cfg = cfg['input']['ssm']
    (
        ssm_lt_sources,
        ssm_lt_weights,
        ssm_lt_rup_counts,
        gsim_lt,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        test_dir / source_cfg["ssm_dir"],
        )

    assert tuple(ssm_lt_sources.keys()) == (0,1)
    assert len(ssm_lt_sources[0]) == 2
    assert ssm_lt_sources[0][0].__class__.__name__ == 'PointSource'

    assert ssm_lt_weights == {0: 0.75, 1: 0.25}
    assert ssm_lt_rup_counts == {0: [1, 1], 1: [1, 1]}
    # no need to test gsim_lt

@unittest.skip("not implemented correctly")
def test_2_branches_collapse():
    pass


def test_2_branches_1_branch():
    test_dir = (BASE_PATH / '..' / '..' / 'tests' / 'data' / 'source_models' / 
                '2_branches')
    cfg = read_yaml_config(test_dir / 'test_2_ssm_branches.yaml')
    source_cfg = cfg['input']['ssm']
    source_cfg['branch'] = 1
    (
        ssm_lt_sources,
        ssm_lt_weights,
        ssm_lt_rup_counts,
        gsim_lt,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        test_dir / source_cfg["ssm_dir"],
        branch=source_cfg['branch'],
        )

    assert tuple(ssm_lt_sources.keys()) == (1,)
    assert len(ssm_lt_sources[1]) == 2
    assert ssm_lt_weights == {1: 1.0}
    assert ssm_lt_rup_counts == {1: [1,1]}
