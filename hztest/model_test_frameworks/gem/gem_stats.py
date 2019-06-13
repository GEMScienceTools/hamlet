from typing import Union, Dict

import numpy as np

from hztest.utils.stats import poisson_likelihood


def calc_mag_bin_empirical_likelihood(bin_rate: Dict[int, float],
                                      n_eqs: int,
                                      not_modeled_val: float = 1e-5) -> float:
    """
    Calculation of the likelihood of observing a certain number of earthquakes
    in a magnitude bin based on an empirical distribution of earthquake
    occurrences, e.g. as aggregated from many iterations a Monte Carlo
    simulation, or another instance in which earthquake occurrence may not be
    Poissonian.

    :param bin_rate:
        the `bin_rate` parameter will be a dictionary, with integer
        keys representing the number of earthquakes and floating point values
        representing the rates of occurrence of each of those integers.

    :param n_eqs:
        Number of observed earthquakes in the magnitude bin (or, number of
        earthquakes for which one wants to calculate the likelihood of seeing in
        that bin).

    :param not_modeled_val:
        Likelihood value resulting from an instance in which no ruptures are
        present in the magnitude bin, but an earthquake has been observed. This
        value should be low, but because the total spatial bin likelihood is the
        product of each of the magnitude bin likelihoods, a value of zero for a
        single magnitude bin (which could be due to rounding or uncertainty in 
        an earthquake catalog, or some coarseness in source model construction) 
        will propagate create a likelihood of zero for the whole spatial bin and
        perhaps the model, depending on assumptions.

    :returns:
        Likelihood of observing a certain number of earthquakes given a rate of
        earthquake occurrence.

    """
    try:
        return bin_rate[n_eqs]
    except KeyError:
        return not_modeled_val


def calc_mag_bin_poisson_likelihood(bin_rate: float,
                                    n_eqs: int,
                                    not_modeled_val: float = 1e-5) -> float:
    """
    Calculation of the likelihood of observing a certain number of earthquakes
    using a poisson calculation.

    :param bin_rate:
        Mean rate of earthquake occurrence within a magnitude bin.

    :param n_eqs:
        Number of observed earthquakes in the magnitude bin (or, number of
        earthquakes for which one wants to calculate the likelihood of seeing in
        that bin).

    :param not_modeled_val:
        Likelihood value resulting from an instance in which no ruptures are
        present in the magnitude bin, but an earthquake has been observed. This
        value should be low, but because the total spatial bin likelihood is the
        product of each of the magnitude bin likelihoods, a value of zero for a
        single magnitude bin (which could be due to rounding or uncertainty in 
        an earthquake catalog, or some coarseness in source model construction) 
        will propagate create a likelihood of zero for the whole spatial bin and
        perhaps the model, depending on assumptions.

    :returns:
        Likelihood of observing a certain number of earthquakes given a rate of
        earthquake occurrence.

    """

    return poisson_likelihood(bin_rate, n_eqs, not_modeled_val)


def calc_mag_bin_likelihood(bin_rate: Union[dict, float],
                            n_eqs: int,
                            not_modeled_val: float = 1e-5,
                            likelihood_method='empirical') -> float:
    """
    Shell function to calculate the likelihood of a magnitude bin given the
    observed earthquakes within the bin. Several methods exist for these
    calculations, though only one is currently implemented.

    :param bin_rate:
        Expected rate(s) of earthquakes within that magnitude bin. The format of
        the rate will depend on the type of analysis: for an empirical
        calculation, the `bin_rate` parameter will be a dictionary, with integer
        keys representing the number of earthquakes and floating point values
        representing the rates of occurrence of each of those integers. For a
        poisson analysis, this should be a float representing the mean rate.

    :param n_eqs:
        Number of observed earthquakes in the magnitude bin (or, number of
        earthquakes for which one wants to calculate the likelihood of seeing in
        that bin).

    :param not_modeled_val:
        Likelihood value resulting from an instance in which no ruptures are
        present in the magnitude bin, but an earthquake has been observed. This
        value should be low, but because the total spatial bin likelihood is the
        product of each of the magnitude bin likelihoods, a value of zero for a
        single magnitude bin (which could be due to rounding or uncertainty in 
        an earthquake catalog, or some coarseness in source model construction) 
        will propagate create a likelihood of zero for the whole spatial bin and
        perhaps the model, depending on assumptions.

    :param likelihood_method:
        Type of analysis that is to be done. Possible values are `empirical`,
        which uses empirical MFDs derived from e.g. Monte Carlo simulations, and
        `poisson`, which uses a single mean rate for each magnitude bin.

    :returns:
        Likelihood of observing a certain number of earthquakes given a rate of
        earthquake occurrence.
    """
    if likelihood_method == 'empirical':
        return calc_mag_bin_empirical_likelihood(bin_rate, n_eqs,
                                                 not_modeled_val)

    elif likelihood_method == 'poisson':
        return calc_mag_bin_poisson_likelihood(bin_rate, n_eqs,
                                               not_modeled_val)


def calc_mfd_log_likelihood_independent(obs_eqs: dict,
                                        mfd: dict,
                                        not_modeled_val: float = 1e-5,
                                        likelihood_method='empirical'
                                        ) -> float:
    """
    Calculation of the log-likelihood of observing earthquakes of a range of
    sizes in a spatial bin (containing many magnitude bins).

    """
    n_bins: int = len(obs_eqs.keys())

    bin_likes = [
        calc_mag_bin_likelihood(mfd[bin_center], len(eqs), not_modeled_val,
                                likelihood_method)
        for bin_center, eqs in obs_eqs.items()
    ]

    return np.exp(np.sum(np.log(bin_likes)) / n_bins)
