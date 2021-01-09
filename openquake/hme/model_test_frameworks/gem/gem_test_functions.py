"""
Utility functions for running tests in the GEM model test framework.
"""
from typing import Sequence, Dict, List, Optional

import numpy as np
from geopandas import GeoSeries

from openquake.hme.utils import SpacemagBin, parallelize, mag_to_mo


def get_mfd_freq_counts(eq_counts: Sequence[int]) -> Dict:
    """
    Makes a dictionary of frequencies of observing different numbers of
    earthquakes, given an input sequence . Keys are the number of earthquakes,
    values are the frequencies.

    >>> get_mfd_freq_counts([4, 4, 5, 2, 4, 2, 5, 6])
    {3: 0.25, 4: 0.375, 5: 0.25, 6: 0.125}

    :param eq_counts:
        Sequence of earthquake counts.

    :returns:
        Dictionary of earthquake count frequencies.

    """

    n_eqs, n_occurrences = np.unique(eq_counts, return_counts=True)
    mfd_freq_counts = {
        n_eq: n_occurrences[i] / sum(n_occurrences)
        for i, n_eq in enumerate(n_eqs)
    }
    return mfd_freq_counts


def get_stochastic_mfd_counts(
        spacemag_bin: SpacemagBin, n_iters: int,
        interval_length: float) -> Dict[float, List[int]]:
    """
    Builds a dictionary of stochastic earthquake counts from a SpacemagBin by
    iteratively sampling the bin and recording the number of events of each
    magnitude.

    :param spacemag_bin:
        The :class:`SpacemagBin` instance to be studied

    :param n_iters:
        Number of Monte Carlo iterations

    :param interval_length:
        Duration of each Monte Carlo sampling iteration (i.e., how many years of
        seismicity to generate in each iteration).

    :returns:
        Dictionary with keys of earthquake magnitudes and values of a list of
        number of occurrences (counts) of that magnitude for each iteration.
    """

    mfd_counts: Dict[float,
                     list] = {bc: []
                              for bc in spacemag_bin.mag_bin_centers}

    for i in range(n_iters):
        for bc, n_eqs in spacemag_bin.get_rupture_sample_mfd(
                interval_length, normalize=False, cumulative=False).items():
            mfd_counts[bc].append(int(n_eqs))

    return mfd_counts


def get_stochastic_mfd(
        spacemag_bin: SpacemagBin, n_iters: int,
        interval_length: float) -> Dict[float, Dict[float, float]]:
    """
    Builds an empirical, incremental magnitude-frequency distribution by
    stochastically sampling earthquake ruptures from a :class:`SpacemagBin`
    instance iteratively. The resulting MFD has the empirical frequencies of the
    number of earthquake occurrences for each magnitude bin over the time
    interval provided.

    :param spacemag_bin: The :class:`SpacemagBin` instance to be studied

    :param n_iters: Number of Monte Carlo iterations

    :param interval_length: Duration of each Monte Carlo sampling iteration
        (i.e., how many years of seismicity to generate in each iteration).

    :returns: Dictionary with keys of earthquake magnitudes and values of a list
        of number of occurrences (counts) of that magnitude for each iteration.
    """
    mfd_counts = get_stochastic_mfd_counts(spacemag_bin, n_iters,
                                           interval_length)

    mfd_freq_counts = {}

    for bc, eq_counts in mfd_counts.items():
        mfd_freq_counts[bc] = get_mfd_freq_counts(eq_counts)

    return mfd_freq_counts


def _source_bin_mfd_apply(geo_series: GeoSeries, **kwargs):
    return geo_series.apply(get_stochastic_mfd, **kwargs)


def get_stochastic_mfds_parallel(geo_series: GeoSeries, **kwargs):
    return parallelize(geo_series, _source_bin_mfd_apply, **kwargs)


def get_stochastic_moment_set(spacemag_bin: SpacemagBin,
                              interval_length: float,
                              n_iters: int,
                              rand_seed=None) -> np.ndarray:
    """
    """
    return np.array([
        get_stochastic_moment(spacemag_bin, interval_length)
        for i in range(n_iters)
    ])


def get_stochastic_moment(spacemag_bin: SpacemagBin,
                          interval_length: float, rand_seed=None) -> float:
    """
    """
    stoch_eq_dict = spacemag_bin.sample_ruptures(
        interval_length=interval_length,
        return_rups=True,
        rand_seed=rand_seed
    )
    moment_sum = np.sum([np.sum([mag_to_mo(rup.magnitude) for rup in bin_eqs])
                     for bin_eqs in stoch_eq_dict.values()])
    return moment_sum


def rank_obs_moment(spacemag_bin: SpacemagBin, interval_length: float, 
                    n_iters: int, stoch_moment_sums: Optional[np.ndarray]) -> float:
    """
    """
    obs_moment_sum = np.sum([np.sum([mag_to_mo(rup.magnitude) for rup in bin_eqs])
                        for bin_eqs in spacemag_bin.observed_earthquakes.values()])
                        
    if stoch_moment_sums is None:
        stoch_moment_sums = get_stochastic_moment_set(spacemag_bin, 
                                                     interval_length,
                                                     n_iters)
                                                     
    n_less = len(stoch_moment_sums[stoch_moment_sums < obs_moment_sum])
    return n_less / n_iters
    

def obs_stoch_moment_ratio(spacemag_bin: SpacemagBin, interval_length: float, 
                    n_iters: int, stoch_moment_sums: Optional[np.ndarray]) -> float:
    """
    """
    obs_moment_sum = np.sum([np.sum([mag_to_mo(rup.magnitude) for rup in bin_eqs])
                        for bin_eqs in spacemag_bin.observed_earthquakes.values()])
                        
    if stoch_moment_sums is None:
        stoch_moment_sums = get_stochastic_moment_set(spacemag_bin, 
                                                     interval_length,
                                                     n_iters)
                                                     
    stoch_moment_mean = np.mean(stoch_moment_sums)

    return obs_moment_sum / stoch_moment_mean


def eval_obs_moment(spacemag_bin: SpacemagBin, interval_length: float,
                    n_iters: int) -> dict:
    """
    """

    stoch_moment_sums = get_stochastic_moment_set(spacemag_bin, interval_length,
                                                  n_iters)

    obs_moment_rank = rank_obs_moment(spacemag_bin, interval_length, n_iters,
                                      stoch_moment_sums=stoch_moment_sums)

    moment_ratio = obs_stoch_moment_ratio(spacemag_bin, interval_length, n_iters,
                                      stoch_moment_sums=stoch_moment_sums)

    return {"obs_moment_rank": obs_moment_rank, "moment_ratio": moment_ratio}
    

def eval_obs_moment_model(spacemag_bins, interval_length: float, n_iters: int):

    cell_stoch_moment_sums = np.array([get_stochastic_moment_set(spacemag_bin, interval_length, 
                                                  n_iters)
                        for spacemag_bin in spacemag_bins])

    obs_moment_sums = np.array([np.sum([np.sum([mag_to_mo(rup.magnitude) for rup in bin_eqs])
                        for bin_eqs in spacemag_bin.observed_earthquakes.values()])
                for spacemag_bin in spacemag_bins])

    model_stoch_moment_sums = cell_stoch_moment_sums.sum(axis=0)
    model_stoch_moment_mean = np.mean(model_stoch_moment_sums)

    obs_moment_sum = obs_moment_sums.sum()

    obs_moment_rank = len(
        model_stoch_moment_sums[model_stoch_moment_sums < obs_moment_sum]
        ) / n_iters

    moment_ratio = obs_moment_sum / model_stoch_moment_mean

    return {"model_obs_moment_rank": obs_moment_rank, 
            "model_moment_ratio": moment_ratio,
            "stoch_moment_sums": model_stoch_moment_sums}


def get_moment_from_mfd(mfd: dict) -> float:
    mo = sum(
        mag_to_mo(np.array(list(mfd.keys()))) * np.array(list(mfd.values())))

    return mo
