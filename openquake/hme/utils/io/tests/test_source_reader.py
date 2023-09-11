import unittest

from openquake.hazardlib.source import SimpleFaultSource

from openquake.hme.utils.tests import load_sm1
from openquake.hme.utils.io.source_reader import (
    csm_from_job_ini,
    get_rlz_source,
    get_csm_rlzs,
    process_source_logic_tree_oq,
    make_composite_source,
    # get_branch_weights,
    make_job_ini,
)


source_cfg = load_sm1.cfg["input"]["ssm"]

# w/ job ini
# get job ini
# get csm
# get rlzs
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
            # "source_model_logic_tree": "/Users/itchy/src/gem/hamlet/openquake/hme/utils/tests/data/source_models/sm1/ssmLT.xml",
            # "gsim_logic_tree": "/Users/itchy/src/gem/hamlet/openquake/hme/utils/tests/data/source_models/sm1/gmmLT.xml",
            "source_model_logic_tree": "ssmLT.xml",
            "gsim_logic_tree": "gmmLT.xml",
            "reference_vs30_type": "measured",
            "reference_vs30_value": "800.0",
            "reference_depth_to_1pt0km_per_sec": "30.0",
            "truncation_level": "3.0",
            "job_ini": "<in-memory>",
            "inputs": {
                # "source_model_logic_tree": "/Users/itchy/src/gem/hamlet/openquake/hme/utils/tests/data/source_models/sm1/ssmLT.xml"
                "source_model_logic_tree": "/ssmLT.xml"
            },
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
        csm, sources, source_info = csm_from_job_ini(job_ini)

        assert csm.code == {"88": b"S"}
        # not really sure what can be easily checked, because
        # the csm is made of a bunch of OQ classes that I don't
        # want try to to store and load

        return csm, sources, source_info

    csm, _sources, _source_info = test_csm_from_job_ini()

    def test_get_rlz_source():
        srcs = get_rlz_source(0, csm)

        assert len(srcs) == 18
        assert isinstance(srcs[0], SimpleFaultSource)

    test_get_rlz_source()

    def test_get_csm_rlzs():
        csm_rlz_groups = get_csm_rlzs(csm)

        assert list(csm_rlz_groups.keys()) == [0]
        assert csm_rlz_groups[0]["weight"] == 1.0
        assert len(csm_rlz_groups[0]["sources"]) == 18

        return csm_rlz_groups

    csm_rlz_groups = test_get_csm_rlzs()

    assert 1 == 1


def test_process_source_logic_tree_oq():
    (
        ssm_lt_sources,
        ssm_lt_weights,
        ssm_lt_rup_counts,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        source_cfg["ssm_dir"],
        lt_file=source_cfg["ssm_lt_file"],
        source_types=source_cfg["source_types"],
        tectonic_region_types=source_cfg["tectonic_region_types"],
        branch=source_cfg["branch"],
        description=load_sm1.cfg["meta"]["description"],
    )

    assert list(ssm_lt_sources.keys()) == ["composite"]
    assert len(ssm_lt_sources["composite"]) == 18
    assert isinstance(ssm_lt_sources["composite"][0], SimpleFaultSource)

    assert list(ssm_lt_weights.keys()) == ["composite"]
    assert len(ssm_lt_weights["composite"]) == 7797
    assert ssm_lt_weights["composite"][0] == 1.0
    assert sum(ssm_lt_weights["composite"]) == 7797.0
