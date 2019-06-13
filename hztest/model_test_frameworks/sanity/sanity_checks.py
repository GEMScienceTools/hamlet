import logging
from typing import Optional

from geopandas import GeoDataFrame, GeoSeries

from .sanity_test_functions import (check_bin_max, )

def min_max_check():
    raise NotImplementedError

def max_check(bin_gdf: GeoDataFrame, append_check: bool=False, 
              warn: bool=False) -> None:
    
    max_check_col: GeoSeries = bin_gdf.apply(check_bin_max)

    if warn is True:
        for i, mxc in max_check_col.iteritems():
            if mxc is False:
                logging.warn('bin {} fails max mag test.'.format(i))

    if append_check is True:
        bin_gdf['max_check'] = max_check_col



def min_check():
    # should check source ruptures and min bin config, maybe?
    raise NotImplementedError


sanity_test_dict = {'min_max': min_max_check}