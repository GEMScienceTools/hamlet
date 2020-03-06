import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils import get_source_bins
from openquake.hme.utils.plots import plot_mfd
from ..sanity.sanity_checks import max_check
from .gem_test_functions import (get_stochastic_mfd,
                                 get_stochastic_mfds_parallel)
from .gem_stats import calc_mfd_log_likelihood_independent


def mfd_likelihood_test(cfg,
                        bin_gdf: Optional[GeoDataFrame] = None,
                        obs_seis_catalog: Optional[GeoDataFrame] = None,
                        pro_seis_catalog: Optional[GeoDataFrame] = None,
                        validate: bool = False):
    """
    Calculates the likelihood of the Seismic Source Model for each SpacemagBin.
    The likelihood calculation is (currently) treated as the geometric mean of
    the individual MagBin likelihoods, which themselves are the likelihood of
    observing the number of earthquakes within that spatial-magnitude bin given
    the total modeled earthquake rate (from all sources) within the
    spatial-magnitude bin.

    The likelihood calculation may be done using the Poisson distribution, if
    there is a basic assumption of Poissonian seismicity, or through a Monte
    Carlo-based calculation (currently also done through a Poisson sampling t_yrs though more complex temporal occurrence models are
    possible, such as through a Epidemic-Type Aftershock Sequence).
    """

    logging.info('Running GEM MFD Likelihood Test')
    if cfg['config']['model_framework']['gem']['likelihood'][
            'likelihood_method'] == 'empirical':
        mfd_empirical_likelihood_test(cfg, bin_gdf, obs_seis_catalog, validate)
    elif cfg['config']['model_framework']['gem']['likelihood'][
            'likelihood_method'] == 'poisson':
        mfd_poisson_likelihood_test(cfg, bin_gdf, obs_seis_catalog, validate)

    if 'report' in cfg.keys():
        return bin_gdf.log_like.describe().to_frame().to_html()


def mfd_empirical_likelihood_test(
        cfg,
        bin_gdf: Optional[GeoDataFrame] = None,
        obs_seis_catalog: Optional[GeoDataFrame] = None,
        pro_seis_catalog: Optional[GeoDataFrame] = None,
        validate: bool = False) -> None:
    """
    Calculates the (log)likelihood of observing the earthquakes in the seismic
    catalog in each :class:`~openquake.hme.utils.bins.SpacemagBin` as the
    geometric mean of the Poisson likelihoods of observing the earthquakes
    within each :class:`~openquake.hme.utils.bins.MagBin` of the
    :class:`~openquake.hme.utils.bins.SpacemagBin`.

    The likelihoods are calculated using the empirical likelihood of observing
    the number of events that occurred in each
    :class:`~openquake.hme.utils.bins.MagBin` given the occurrence rate for 
    that :class:`~openquake.hme.utils.bins.MagBin`. This is done through a Monte
    Carlo simulation, which returns the fraction of the total Monte Carlo
    samples had the same number of events as observed.

    The likelihoods for each :class:`~openquake.hme.utils.bins.SpacemagBin` are
    then log-transformed and appended as a new column to the `bin_gdf`
    :class:`GeoDataFrame` hosting the bins.
    """

    test_config = cfg['config']['model_framework']['gem']['likelihood']
    source_bin_gdf = get_source_bins(bin_gdf)

    logging.info('calculating empirical MFDs for source bins')

    if cfg['config']['parallel'] is False:
        source_bin_mfds = source_bin_gdf['SpacemagBin'].apply(
            get_stochastic_mfd,
            n_iters=test_config['n_iters'],
            interval_length=test_config['investigation_time'])
    else:
        source_bin_mfds = get_stochastic_mfds_parallel(
            source_bin_gdf['SpacemagBin'],
            n_iters=test_config['n_iters'],
            interval_length=test_config['investigation_time'])

    def calc_row_log_like(row, mfd_df=source_bin_mfds):
        obs_eqs = row.SpacemagBin.observed_earthquakes
        mfd_dict = mfd_df.loc[row._name]

        return calc_mfd_log_likelihood_independent(
            obs_eqs,
            mfd_dict,
            not_modeled_val=test_config['not_modeled_val'],
            likelihood_method='empirical')

    logging.info('calculating log likelihoods for sources')
    source_bin_log_likes = source_bin_gdf.apply(calc_row_log_like, axis=1)

    bin_gdf['log_like'] = test_config['default_likelihood']
    bin_gdf['log_like'].update(source_bin_log_likes)


def mfd_poisson_likelihood_test(
        cfg,
        bin_gdf: Optional[GeoDataFrame] = None,
        obs_seis_catalog: Optional[GeoDataFrame] = None,
        pro_seis_catalog: Optional[GeoDataFrame] = None,
        validate: bool = False) -> None:
    """
    Calculates the (log)likelihood of observing the earthquakes in the seismic
    catalog in each :class:`~openquake.hme.utils.bins.SpacemagBin` as the
    geometric mean of the Poisson likelihoods of observing the earthquakes
    within each :class:`~openquake.hme.utils.bins.MagBin` of the
    :class:`~openquake.hme.utils.bins.SpacemagBin`.

    The likelihoods are calculated using the Poisson likelihood of observing
    the number of events that occurred in each
    :class:`~openquake.hme.utils.bins.MagBin` given the occurrence rate for 
    that :class:`~openquake.hme.utils.bins.MagBin`.  See
    :func:`~openquake.hme.utils.stats.poisson_likelihood` for more information.

    The likelihoods for each :class:`~openquake.hme.utils.bins.SpacemagBin` are
    then log-transformed and appended as a new column to the `bin_gdf`
    :class:`GeoDataFrame` hosting the bins.
    """

    test_config = cfg['config']['model_framework']['gem']['likelihood']
    source_bin_gdf = get_source_bins(bin_gdf)

    logging.info('calculating empirical MFDs for source bins')

    source_bin_mfds = source_bin_gdf['SpacemagBin'].apply(
        lambda x: x.get_rupture_mfd(cumulative=False))

    def calc_row_log_like(row, mfd_df=source_bin_mfds):
        obs_eqs = row.SpacemagBin.observed_earthquakes
        mfd_dict = mfd_df.loc[row._name]

        return calc_mfd_log_likelihood_independent(
            obs_eqs,
            mfd_dict,
            time_interval=test_config['investigation_time'],
            not_modeled_val=test_config['not_modeled_val'],
            likelihood_method='poisson')

    logging.info('calculating log likelihoods for sources')
    source_bin_log_likes = source_bin_gdf.apply(calc_row_log_like, axis=1)

    bin_gdf['log_like'] = test_config['default_likelihood']
    bin_gdf['log_like'].update(source_bin_log_likes)


def model_mfd_test(cfg,
                   bin_gdf: Optional[GeoDataFrame] = None,
                   obs_seis_catalog: Optional[GeoDataFrame] = None,
                   pro_seis_catalog: Optional[GeoDataFrame] = None,
                   validate: bool = False) -> None:

    # calculate observed, model mfd for all bins
    # add together

    logging.info('Running Model-Observed MFD Comparison')

    test_config = cfg['config']['model_framework']['gem']['model_mfd']

    mod_mfd = bin_gdf.iloc[0].SpacemagBin.get_rupture_mfd()
    obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_empirical_mfd(
        t_yrs=test_config['investigation_time'])

    for i, row in bin_gdf.iloc[1:].iterrows():
        bin_mod_mfd = row.SpacemagBin.get_rupture_mfd()
        bin_obs_mfd = row.SpacemagBin.get_empirical_mfd(
            t_yrs=test_config['investigation_time'])

        for bin_center, rate in bin_mod_mfd.items():
            mod_mfd[bin_center] += rate

        for bin_center, rate in bin_obs_mfd.items():
            obs_mfd[bin_center] += rate

    mfd_df = pd.DataFrame.from_dict(mod_mfd,
                                    orient='index',
                                    columns=['mod_mfd'])
    mfd_df['mod_mfd_cum'] = np.cumsum(mfd_df['mod_mfd'].values[::-1])[::-1]

    mfd_df['obs_mfd'] = obs_mfd.values()
    mfd_df['obs_mfd_cum'] = np.cumsum(mfd_df['obs_mfd'].values[::-1])[::-1]

    mfd_df.index.name = 'bin'

    # refactor below -- this shouldn't be in test_config
    if 'out_csv' in test_config.keys():
        mfd_df.to_csv(test_config['out_csv'])

    if 'out_plot' in test_config.keys():
        plot_mfd(model=mfd_df['mod_mfd_cum'].to_dict(),
                 observed=mfd_df['obs_mfd_cum'].to_dict(),
                 save_fig=test_config['out_plot'])

    if 'report' in cfg.keys():
        return plot_mfd(model=mfd_df['mod_mfd_cum'].to_dict(),
                        observed=mfd_df['obs_mfd_cum'].to_dict(),
                        t_yrs=test_config['investigation_time'],
                        return_fig=False,
                        return_string=True)


def max_mag_check(cfg: dict,
                  bin_gdf: Optional[GeoDataFrame] = None,
                  obs_seis_catalog: Optional[GeoDataFrame] = None,
                  pro_seis_catalog: Optional[GeoDataFrame] = None):

    logging.info('Checking Maximum Magnitudes')

    test_config = cfg['config']['model_framework']['gem']['max_mag_check']

    bad_bins = max_check(bin_gdf, append_check=True, warn=test_config['warn'])

    if 'report' in cfg.keys():
        return bad_bins


gem_test_dict = {
    'likelihood': mfd_likelihood_test,
    'max_mag_check': max_mag_check,
    'model_mfd': model_mfd_test
}
