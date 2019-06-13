import logging
from typing import Optional

from geopandas import GeoDataFrame

#from hztest.core.core import make_bin_gdf

from hztest.utils import get_source_bins
from .gem_test_functions import get_stochastic_mfd
from .gem_stats import calc_mfd_log_likelihood_independent



def mfd_likelihood_test(cfg, bin_gdf: Optional[GeoDataFrame]=None, 
                        obs_seis_catalog: Optional[GeoDataFrame]=None, 
                        validate: bool=False):

    """


    """

    logging.info('Running GEM MFD Likelihood Test')
    if cfg['config']['tests']['likelihood']['likelihood_method'] == 'empirical':
        return mfd_empirical_likelihood_test(cfg, bin_gdf, 
                                             obs_seis_catalog, validate)



def mfd_empirical_likelihood_test(cfg, bin_gdf: Optional[GeoDataFrame]=None, 
                                  obs_seis_catalog: Optional[GeoDataFrame]=None, 
                                  validate: bool=False):

    test_config = cfg['config']['tests']['likelihood']
    source_bin_gdf = get_source_bins(bin_gdf)

    logging.info('calculating empirical MFDs for source bins')
    source_bin_mfds = source_bin_gdf['SpacemagBin'].apply(get_stochastic_mfd,
               n_iters=test_config['n_iters'],
               interval_length=test_config['investigation_time'])

    def calc_row_log_like(row, mfd_df=source_bin_mfds):
        obs_eqs = row.SpacemagBin.observed_earthquakes
        mfd_dict = mfd_df.loc[row._name]

        return calc_mfd_log_likelihood_independent(obs_eqs, mfd_dict,
            not_modeled_val=test_config['not_modeled_val'],
            likelihood_method='empirical')

    logging.info('calculating log likelihoods for sources')
    source_bin_log_likes = source_bin_gdf.apply(calc_row_log_like, axis=1)

    bin_gdf['log_like'] = test_config['default_likelihood']
    bin_gdf['log_like'].update(source_bin_log_likes)




class MFDLikelihoodTest():
    """
    Do I even want this?
    """
    def __init__(self):
        self.cfg = None
        self.bin_df = None
        self.obs_seis_catalog = None

        raise NotImplementedError

    def run_test(self):
        return mfd_likelihood_test(self.cfg, self.bin_df, self.obs_seis_catalog)



gem_test_dict = {'likelihood': mfd_likelihood_test}