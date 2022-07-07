import logging
from typing import Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils import get_source_bins, get_mag_bins_from_cfg
from openquake.hme.utils.plots import plot_mfd
from ..sanity.sanity_checks import max_check
from .gem_test_functions import (
    get_stochastic_mfd,
    get_stochastic_mfds_parallel,
    eval_obs_moment,
    eval_obs_moment_model,
)

from ..relm.relm_tests import (
    n_test_function,
    s_test_function,
    m_test_function,
    l_test_function,
)
from ..relm.relm_test_functions import m_test_function, s_test_function

from .gem_stats import calc_mfd_log_likelihood_independent


def M_test(
    cfg,
    input_data,
) -> dict:
    """
    The M-Test is based on Zechar et al. (2010), though not identical. This
    tests evaluates the consistency of the magnitude-frequency distribution of
    the model vs. the observations, by evaluating the log-likelihood of the
    observed earthquakes given the model (forecast), compared with the
    log-likelihood of a large number of stochastic catalogs generated from the
    same forecast. If the log-likelihood of the observed earthquake catalog is
    less than the majority of the log-likelihoods of stochastic catalogs
    (specified by the `critical_pct` argument), then the test fails.

    The log-likelihoods are calculated first for each magnitude bin. The
    log-likelihood for each magnitude bin is the log-likelihood of the observed
    (or stochastic) number of earthquakes in that magnitude bin occurring
    throughout the model domain, given the mean rupture rate for that magnitude
    bin, using the Poisson distribution.

    Then, the log-likelihoods of the observed catalog and the stochastic
    catalogs are calculated as the geometric mean of the individual bin
    likelihoods.

    The differences between this implementation and that of Zechar et al. (2010)
    is that 1) in this version we do not fix the total number of earthquakes
    that occurs in each stochastic simulation (because that is somewhat
    complicated to implement within Hamlet) and 2) we use the geometric mean
    instead of the product of the magnitude bin likelihoods for the total
    likelihood, because this lets us disregard the discretization of the MFD
    when comparing between different models. Note that in terms of passing or
    failing, (1) does not matter much if the model passes the N-test, and (2)
    does not matter at all because the ranking of the observed and stochasitc
    catalogs will remain the same.
    """
    logging.info("Running GEM M-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["M_test"]

    prospective = test_config.get("prospective", False)
    critical_pct = test_config.get("critical_pct", 0.25)
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_result = m_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs,
        test_config["n_iters"],
        not_modeled_likelihood=not_modeled_likelihood,
        critical_pct=critical_pct,
    )

    logging.info("M-Test crit pct {}".format(test_result["critical_pct"]))
    logging.info("M-Test pct {}".format(test_result["percentile"]))
    logging.info("M-Test {}".format(test_result["test_res"]))
    return test_result


def S_test(
    cfg: dict,
    input_data: dict,
) -> dict:
    """"""
    logging.info("Running GEM S-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["S_test"]
    prospective = test_config.get("prospective", False)
    likelihood_function = test_config.get("likelihood_function", "mfd")
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    test_config["parallel"] = cfg["config"]["parallel"]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_results = s_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        likelihood_function,
        mag_bins=mag_bins,
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("S-Test {}".format(test_results["test_res"]))
    logging.info("S-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("S-Test model pct: {}".format(test_results["percentile"]))
    return test_results


def L_test(
    cfg: dict,
    input_data: dict,
) -> dict:
    """"""
    logging.info("Running GEM L-Test")

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["L_test"]
    prospective = test_config.get("prospective", False)
    not_modeled_likelihood = 0.0  # hardcoded for RELM
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
        eq_groups = input_data["pro_groups"]
        t_yrs = test_config["investigation_time"]
    else:
        eq_gdf = input_data["eq_gdf"]
        eq_groups = input_data["eq_groups"]
        t_yrs = cfg["input"]["seis_catalog"]["duration"]

    test_results = l_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        input_data["cell_groups"],
        eq_groups,
        t_yrs,
        test_config["n_iters"],
        mag_bins,
        critical_pct=test_config["critical_pct"],
        not_modeled_likelihood=not_modeled_likelihood,
    )

    logging.info("L-Test {}".format(test_results["test_res"]))
    logging.info("L-Test crit pct: {}".format(test_results["critical_pct"]))
    logging.info("L-Test model pct: {}".format(test_results["percentile"]))
    return test_results


def N_test(cfg: dict, input_data: dict) -> dict:
    logging.info("Running N-Test")

    test_config = cfg["config"]["model_framework"]["gem"]["N_test"]

    prospective = test_config.get("prospective", False)

    if (test_config["prob_model"] == "poisson") and not prospective:
        test_config["investigation_time"] = cfg["input"]["seis_catalog"][
            "duration"
        ]

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    test_results = n_test_function(
        input_data["rupture_gdf"], eq_gdf, test_config
    )

    logging.info(
        "N-Test number obs eqs: {}".format(test_results["n_obs_earthquakes"])
    )
    logging.info(
        "N-Test number pred eqs: {}".format(test_results["inv_time_rate"])
    )
    logging.info("N-Test {}".format(test_results["test_pass"]))
    return test_results


def max_mag_check(cfg: dict, input_data: dict):

    logging.info("Checking Maximum Magnitudes")

    max_bin_check_results = max_check(cfg, input_data, framework="gem")

    bad_bins = [
        cell
        for cell, max_check_val in max_bin_check_results.items()
        if max_check_val is False
    ]

    # could add all results here for the map...
    if bad_bins == []:
        results = {"test_res": "Pass", "test_pass": True, "bad_bins": bad_bins}
    else:
        results = {"test_res": "Fail", "test_pass": False, "bad_bins": bad_bins}

    logging.info("Max Mag Check res: {}".format(results["test_res"]))
    return results


def model_mfd_eval():
    raise NotImplementedError()


def moment_over_under_eval():
    pass


##########
# old tests
##########


def moment_over_under_eval_old(cfg: dict, bin_gdf: GeoDataFrame):
    """
    The Moment Over-Under evaluation compares each cell's total seismic moment
    forecast by the model to the observed moment release from the earthquake
    catalog. The evaluation uses stochastic event sets so that the catalog is
    more directly comparable to the forecast.
    """

    logging.info("Running Over-Under evaluation")

    test_config = cfg["config"]["model_framework"]["gem"]["moment_over_under"]

    # these two tests can be combined by accessing the stochastic eqs
    # in the bins -- this should be done for many tests.

    # evalutates the bins independently
    # returns a Series of dicts
    obs_moment_evals = bin_gdf.SpacemagBin.apply(
        eval_obs_moment,
        args=(test_config["investigation_time"], test_config["n_iters"]),
    )

    # turns the Series of dicts into a DataFrame
    obs_moment_evals = obs_moment_evals.apply(pd.Series)

    # evaluates the whole model
    model_moment_eval = eval_obs_moment_model(
        bin_gdf.SpacemagBin,
        test_config["investigation_time"],
        test_config["n_iters"],
    )

    bin_gdf["moment_rank_pctile"] = obs_moment_evals["obs_moment_rank"]
    bin_gdf["moment_ratio"] = obs_moment_evals["moment_ratio"]

    logging.info(
        "Observed / mean stochastic moment: {}".format(
            round(model_moment_eval["model_moment_ratio"], 3)
        )
    )

    logging.info(
        "Observed moment release rank (higher rank is more moment): {}".format(
            round(model_moment_eval["model_obs_moment_rank"], 3)
        )
    )

    return model_moment_eval


def model_mfd_test_old(cfg: dict, bin_gdf: GeoDataFrame) -> None:

    # calculate observed, model mfd for all bins
    # add together

    logging.info("Running Model-Observed MFD Comparison")

    test_config = cfg["config"]["model_framework"]["gem"]["model_mfd"]

    mod_mfd = bin_gdf.iloc[0].SpacemagBin.get_rupture_mfd()
    obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_empirical_mfd(
        t_yrs=test_config["investigation_time"]
    )

    for i, row in bin_gdf.iloc[1:].iterrows():
        bin_mod_mfd = row.SpacemagBin.get_rupture_mfd()
        bin_obs_mfd = row.SpacemagBin.get_empirical_mfd(
            t_yrs=test_config["investigation_time"]
        )

        for bin_center, rate in bin_mod_mfd.items():
            mod_mfd[bin_center] += rate

        for bin_center, rate in bin_obs_mfd.items():
            obs_mfd[bin_center] += rate

    mfd_df = pd.DataFrame.from_dict(
        mod_mfd, orient="index", columns=["mod_mfd"]
    )
    mfd_df["mod_mfd_cum"] = np.cumsum(mfd_df["mod_mfd"].values[::-1])[::-1]

    mfd_df["obs_mfd"] = obs_mfd.values()
    mfd_df["obs_mfd_cum"] = np.cumsum(mfd_df["obs_mfd"].values[::-1])[::-1]

    mfd_df.index.name = "bin"

    # refactor below -- this shouldn't be in test_config
    if "out_csv" in test_config.keys():
        mfd_df.to_csv(test_config["out_csv"])

    if "out_plot" in test_config.keys():
        plot_mfd(
            model=mfd_df["mod_mfd_cum"].to_dict(),
            observed=mfd_df["obs_mfd_cum"].to_dict(),
            save_fig=test_config["out_plot"],
        )

    if "report" in cfg.keys():
        return plot_mfd(
            model=mfd_df["mod_mfd_cum"].to_dict(),
            observed=mfd_df["obs_mfd_cum"].to_dict(),
            t_yrs=test_config["investigation_time"],
            return_fig=False,
            return_string=True,
        )


def mfd_likelihood_test(cfg: dict, bin_gdf: GeoDataFrame):
    """
    Calculates the likelihood of the Seismic Source Model for each SpacemagBin.
    The likelihood calculation is (currently) treated as the geometric mean of
    the individual MagBin likelihoods, which themselves are the likelihood of
    observing the number of earthquakes within that spatial-magnitude bin given
    the total modeled earthquake rate (from all sources) within the
    spatial-magnitude bin.

    The likelihood calculation may be done using the Poisson distribution, if
    there is a basic assumption of Poissonian seismicity, or through a Monte
    Carlo-based calculation (currently also done through a Poisson sampling
    t_yrs though more complex temporal occurrence models are
    possible, such as through a Epidemic-Type Aftershock Sequence).
    """
    logging.info("Running GEM MFD Likelihood Test")

    like_config = cfg["config"]["model_framework"]["gem"]["likelihood"]

    if like_config["likelihood_method"] == "empirical":
        mfd_empirical_likelihood_test(cfg, bin_gdf)
    elif like_config["likelihood_method"] == "poisson":
        mfd_poisson_likelihood_test(cfg, bin_gdf)

    total_log_like = np.sum(bin_gdf["log_like"]) / bin_gdf.shape[0]

    # if "report" in cfg.keys():
    #    return bin_gdf.log_like.describe().to_frame().to_html()
    results = bin_gdf.log_like.describe().to_dict()
    results["total_log_likelihood"] = (
        np.sum(bin_gdf["log_like"]) / bin_gdf.shape[0]
    )

    return results


def M_test_old(
    cfg,
    input_data,
) -> dict:
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

    The differences between this implementation and that of Zechar et al. (2010)
    are that 1) in this version we do not fix the total number of earthquakes
    that occurs in each stochastic simulation (because that is somewhat
    complicated to implement within Hamlet) and 2) we use the geometric mean
    instead of the product of the magnitude bin likelihoods for the total
    likelihood, because this lets us disregard the discretization of the MFD
    when comparing between different models. Note that in terms of passing or
    failing, (1) does not matter much if the model passes the N-test, and (2)
    does not matter at all because the ranking of the observed and stochasitc
    catalogs will remain the same.
    """

    mag_bins = get_mag_bins_from_cfg(cfg)
    test_config = cfg["config"]["model_framework"]["gem"]["M_test"]
    prospective = test_config.get("prospective", False)
    critical_pct = test_config.get("critical_pct", 0.25)
    t_yrs = test_config["investigation_time"]
    not_modeled_likelihood = test_config.get("not_modeled_likelihood", 1e-5)

    if prospective:
        eq_gdf = input_data["pro_gdf"]
    else:
        eq_gdf = input_data["eq_gdf"]

    test_result = m_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs,
        test_config["n_iters"],
        not_modeled_likelihood=0.0,
        critical_pct=critical_pct,
    )

    test_result = m_test_function(
        input_data["rupture_gdf"],
        eq_gdf,
        mag_bins,
        t_yrs,
        test_config["n_iters"],
        not_modeled_likelihood=0.0,
        critical_pct=critical_pct,
    )

    bin_gdf["log_like"] = test_config["default_likelihood"]
    bin_gdf["log_like"].update(source_bin_log_likes)


def mfd_poisson_likelihood_test(cfg: dict, bin_gdf: GeoDataFrame) -> None:
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

    test_config = cfg["config"]["model_framework"]["gem"]["likelihood"]
    source_bin_gdf = get_source_bins(bin_gdf)

    logging.info("calculating empirical MFDs for source bins")

    source_bin_mfds = source_bin_gdf["SpacemagBin"].apply(
        lambda x: x.get_rupture_mfd(cumulative=False)
    )

    def calc_row_log_like(row, mfd_df=source_bin_mfds):
        obs_eqs = row.SpacemagBin.observed_earthquakes
        mfd_dict = mfd_df.loc[row._name]

        return calc_mfd_log_likelihood_independent(
            obs_eqs,
            mfd_dict,
            time_interval=test_config["investigation_time"],
            not_modeled_val=test_config["not_modeled_val"],
            likelihood_method="poisson",
        )

    logging.info("calculating log likelihoods for sources")
    source_bin_log_likes = source_bin_gdf.apply(calc_row_log_like, axis=1)

    bin_gdf["log_like"] = test_config["default_likelihood"]
    bin_gdf["log_like"].update(source_bin_log_likes)


gem_test_dict = {
    "likelihood": mfd_likelihood_test,
    "max_mag_check": max_mag_check,
    "model_mfd": model_mfd_eval,
    "moment_over_under": moment_over_under_eval,
    "M_test": M_test,
    "S_test": S_test,
    "N_test": N_test,
    "L_test": L_test,
}
