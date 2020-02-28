import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from geopandas import GeoDataFrame

from openquake.hme.utils import get_source_bins
from openquake.hme.utils.plots import plot_mfd


def L_test():
    #
    raise NotImplementedError


def N_test(cfg: dict,
           bin_gdf: Optional[GeoDataFrame] = None,
           obs_seis_catalog: Optional[GeoDataFrame] = None,
           validate: bool = False):
    """

    """

    test_config = cfg['config']['tests']['N_test']

    if 'conf_interval' not in test_config:
        test_config['conf_interval'] = 0.95

    annual_rup_rate = 0.
    obs_eqs = []
    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin
        min_bin_center = np.min(sb.mag_bin_centers)
        bin_mfd = sb.get_rupture_mfd(cumulative=True)
        annual_rup_rate += bin_mfd[min_bin_center]

        for mb in sb.observed_earthquakes.values():
            obs_eqs.extend(mb)

    test_rup_rate = annual_rup_rate * test_config['investigation_time']

    if test_config['prob_model'] == 'poisson':
        conf_min, conf_max = poisson(test_rup_rate).interval(
            test_config['conf_interval'])

        test_pass = conf_min <= len(obs_eqs) <= conf_max
        test_result = {
            'conf_interval_pct': test_config['conf_interval'],
            'conf_interval': (conf_min, conf_max),
            'inv_time_rate': test_rup_rate,
            'n_obs_earthquakes': len(obs_eqs),
            'pass': test_pass
        }

    elif test_config['prob_model'] == 'neg_binom':
        pass

    return test_result


relm_test_dict = {'L_test': L_test, 'N_test': N_test}