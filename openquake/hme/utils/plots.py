import os
from typing import Union, Optional, Tuple, Sequence, Any

import h3
import numpy as np
import geopandas as gpd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import io
from .stats import sample_event_times_in_interval

natural_earth_countries_file = os.path.join(
    *os.path.split(__file__)[::-1],
    "..",
    "datasets",
    "ne_50m_admin_0_countries.geojson",
)


def _sample_n_events(rate):
    return len(sample_event_times_in_interval(rate, interval_length=1.0))


def _make_stoch_mfds(mfd, iters: int, t_yrs: float = 1.0):

    cum_rates = list(mfd.values())

    incr_rates = []
    for i, rate in enumerate(cum_rates):
        if i < len(cum_rates) - 1:
            incr_rates.append(rate - cum_rates[i + 1])
        elif i == len(cum_rates) - 1:
            incr_rates.append(rate)

    incr_rates = np.array(incr_rates) * t_yrs

    stoch_mfd_vals = []

    for i in range(iters):
        n_event_list = np.array(
            [_sample_n_events(rate) for rate in incr_rates], dtype=float
        )
        n_event_list /= t_yrs
        stoch_mfd_vals.append(np.cumsum(n_event_list[::-1])[::-1])

    return stoch_mfd_vals


def plot_N_test_results(
    N_test_results: dict,
    return_fig: bool = False,
    return_string: bool = False,
    save_fig: Union[bool, str] = False,
):

    if N_test_results["prob_model"] == "poisson":
        fig = plot_poisson_distribution(
            N_e=N_test_results["n_pred_earthquakes"],
            N_o=N_test_results["n_obs_earthquakes"],
            conf_interval=N_test_results["conf_interval"],
        )
    elif N_test_results.get("pred_samples"):
        fig = plot_N_test_empirical(
            N_test_results["pred_samples"],
            N_test_results["n_obs_earthquakes"],
            conf_interval=N_test_results["conf_interval"],
        )
    else:
        return None

    if save_fig is not False:
        fig.savefig(save_fig)

    if return_fig is True:
        return fig

    elif return_string is True:
        plt.switch_backend("svg")
        fig_str = io.StringIO()
        fig.savefig(fig_str, format="svg")
        plt.close(fig)
        fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
        return fig_svg


def plot_N_test_empirical(
    n_expected: Sequence[int],
    n_obs: int,
    conf_interval: Optional[Tuple[int, int]] = None,
):
    fig, ax = plt.subplots()
    plt.hist(
        n_expected,
        bins=20,
        histtype="stepfilled",
        label="Expected number of earthquakes",
        color="C0",
        alpha=0.5,
    )

    plt.axvline(n_obs, color="C1", linestyle="-", label="Observed earthquakes")

    if conf_interval is not None:
        plt.axvspan(
            *conf_interval, color="gray", alpha=0.5, label="Test Pass Interval"
        )

    plt.xlabel("Number of Earthquakes")
    plt.ylabel("Frequency")

    plt.legend(loc="best")

    return fig


def plot_L_test_results(
    results: dict[str, Any],
    return_fig: bool = False,
    return_string: bool = False,
    save_fig: Union[bool, str] = False,
):

    stoch_loglikes = results["test_data"]["stoch_loglike_totals"]
    obs_loglike = results["test_data"]["obs_loglike_total"]
    critical_pct = results["critical_pct"]

    fig, ax = plt.subplots()
    plt.hist(
        stoch_loglikes,
        bins=20,
        histtype="stepfilled",
        label="Modeled log-likelihoods",
        color="C0",
        alpha=0.5,
    )

    plt.axvline(
        obs_loglike, color="C1", linestyle="-", label="Observed log-likelihood"
    )

    plt.axvline(
        np.quantile(stoch_loglikes, critical_pct),
        color="C2",
        linestyle="-",
        label="Critical Fractile",
    )

    plt.xlabel("Log-Likelihood")
    plt.ylabel("Count")

    plt.legend(loc="best")

    if save_fig is not False:
        fig.savefig(save_fig)

    if return_fig is True:
        return fig

    elif return_string is True:
        plt.switch_backend("svg")
        fig_str = io.StringIO()
        fig.savefig(fig_str, format="svg")
        plt.close(fig)
        fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
        return fig_svg


def plot_poisson_distribution(N_e, N_o, conf_interval=None, plot_cdf=False):
    """
    Plots the PDF and CDF of a Poisson distribution with mean N_e
    and draws a vertical line at N_o.

    Parameters:
    - N_e: Expected number of events (mean rate of the Poisson distribution)
    - N_o: Observed number of events
    """
    # Generate a range of numbers around N_e to calculate the PDF and CDF
    x = np.arange(poisson.ppf(0.001, N_e), poisson.ppf(0.999, N_e))

    # Calculate the PDF and CDF
    pdf = poisson.pmf(x, N_e)
    cdf = poisson.cdf(x, N_e)

    fig, ax1 = plt.subplots()

    # Plot PDF
    color = "C0"
    ax1.set_xlabel("n (Number of Events)")
    ax1.set_ylabel("p(N) (PDF)", color=color)
    ax1.plot(x, pdf, color=color, label="PDF")
    ax1.tick_params(axis="y", labelcolor=color)

    if plot_cdf:
        # Create a twin Axes object to plot the CDF
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel(
            "CDF", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(x, cdf, color=color, linestyle="--", label="CDF")
        ax2.tick_params(axis="y", labelcolor=color)

    # Draw a vertical line at N_o
    plt.axvline(N_o, color="C1", linestyle="-", label="Observed earthquakes)")

    if conf_interval is not None:
        plt.axvspan(
            *conf_interval, color="gray", alpha=0.5, label="Test Pass Interval"
        )

    # Add a legend
    lines, labels = ax1.get_legend_handles_labels()
    if plot_cdf:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax1.legend(lines, labels, loc="best")

    plt.title(
        # "Poisson Distribution (PDF and CDF) with N_e = {} and N_o = {}".format(
        #     N_e, N_o
        # )
        "Total number of earthquakes"
    )

    return fig
    # if return_fig is True:
    #    return fig

    # elif return_string is True:
    #    plt.switch_backend("svg")
    #    fig_str = io.StringIO()
    #    fig.savefig(fig_str, format="svg")
    #    plt.close(fig)
    #    fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
    #    return fig_svg


def plot_mfd(
    model: Optional[dict] = None,
    model_format: str = "C0-",
    model_label: str = "model",
    model_iters: int = 500,
    observed: Optional[dict] = None,
    observed_format: str = "C3o-.",
    observed_label: str = "observed",
    t_yrs: float = 1.0,
    return_fig: bool = True,
    return_string: bool = False,
    save_fig: Union[bool, str] = False,
    **kwargs,
):
    """
    Makes a plot of empirical MFDs

    """
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, yscale="log")
    plt.title("Magnitude-Frequency Distribution")

    if model is not None:
        if model_iters > 0:
            stoch_mfd_vals = _make_stoch_mfds(
                model, iters=model_iters, t_yrs=t_yrs
            )
            for smfd in stoch_mfd_vals:
                ax.plot(
                    list(model.keys()), smfd, model_format, lw=10 / model_iters
                )

    if model is not None:
        ax.plot(
            list(model.keys()),
            list(model.values()),
            model_format,
            label=model_label,
        )

    if observed is not None:
        ax.plot(
            list(observed.keys()),
            list(observed.values()),
            observed_format,
            label=observed_label,
        )

    ax.legend(loc="upper right")
    ax.set_ylabel("Annual frequency of exceedance")
    ax.set_xlabel("Magnitude")

    if save_fig is not False:
        fig.savefig(save_fig)

    if return_fig is True:
        return fig

    elif return_string is True:
        plt.switch_backend("svg")
        fig_str = io.StringIO()
        fig.savefig(fig_str, format="svg")
        plt.close(fig)
        fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
        return fig_svg


def plot_likelihood_map(
    bin_gdf: GeoDataFrame,
    plot_eqs: bool = True,
    eq_gdf: Optional[GeoDataFrame] = None,
    map_epsg: Optional[int] = None,
):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if map_epsg is None:
        bin_gdf.plot(
            column="log_like",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="OrRd_r",
            legend=True,
        )
    else:
        bin_gdf.to_crs(epsg=map_epsg).plot(
            column="log_like",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="OrRd_r",
            legend=True,
        )

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(natural_earth_countries_file)
    if map_epsg is None:
        world.plot(ax=ax, color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(
            ax=ax, color="none", edgecolor="black"
        )

    if plot_eqs is True:
        if eq_gdf is not None:
            if map_epsg is None:
                ax.scatter(
                    eq_gdf.longitude,
                    eq_gdf.latitude,
                    s=(eq_gdf.magnitude**3) / 10.0,
                    edgecolor="blue",
                    facecolors="none",
                    alpha=0.3,
                )
            else:
                pass

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.switch_backend("svg")
    fig_str = io.StringIO()
    fig.savefig(fig_str, format="svg")
    fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
    return fig_svg


def plot_S_test_map(
    cell_gdf: GeoDataFrame,
    map_epsg: Optional[int] = None,
    bad_bins: list = list(),
    model_test_framework: str = "gem",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if len(bad_bins) > 0:
        # sometimes at this point the bin_index is no longer the index,
        # not clear why...
        if "bin_index" in cell_gdf.columns:
            bad_bin_gdf = cell_gdf[cell_gdf.bin_index.isin(bad_bins)]
        else:
            bad_bin_gdf = GeoDataFrame(cell_gdf.loc[bad_bins])

    if map_epsg is None:
        cell_gdf.plot(
            column=f"{model_test_framework}_S_test_frac",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="OrRd_r",
            legend=True,
        )
        if len(bad_bins) > 0:
            bad_bin_gdf.plot(ax=ax, color="blue")

    else:
        cell_gdf.to_crs(epsg=map_epsg).plot(
            column=f"{model_test_framework}_S_test_frac",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="OrRd_r",
            legend=True,
        )
        if len(bad_bins) > 0:
            bad_bin_gdf.plot(ax=ax, color="blue")

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(natural_earth_countries_file)
    if map_epsg is None:
        world.plot(ax=ax, color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(
            ax=ax, color="none", edgecolor="black"
        )
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.switch_backend("svg")
    fig_str = io.StringIO()
    fig.savefig(fig_str, format="svg")
    fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
    return fig_svg


def plot_over_under_map(
    cell_gdf: GeoDataFrame, map_epsg: Optional[int] = None
):
    fig, axs = plt.subplots(2, 1, figsize=(10, 18))

    # plot moment ratio

    # get colorbar bounds so that 1 is in the middle
    max_ratio = cell_gdf.moment_over_under_ratio.max()
    min_ratio = cell_gdf.moment_over_under_ratio.min()
    max_r_dist = np.max(np.abs([(1 - max_ratio), (1 - min_ratio)]))

    if map_epsg is None:
        cell_gdf.plot(
            column="moment_over_under_ratio",
            ax=axs[0],
            vmin=1 - max_r_dist,
            vmax=1 + max_r_dist,
            cmap="PRGn",
            legend=True,
        )
    else:
        cell_gdf.to_crs(epsg=map_epsg).plot(
            column="moment_over_under_ratio",
            ax=axs[0],
            vmin=1 - max_r_dist,
            vmax=1 + max_r_dist,
            cmap="PRGn",
            legend=True,
        )

    x_lims = axs[0].get_xlim()
    y_lims = axs[0].get_ylim()

    world = gpd.read_file(natural_earth_countries_file)
    if map_epsg is None:
        world.plot(ax=axs[0], color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(
            ax=axs[0], color="none", edgecolor="black"
        )
    axs[0].set_xlim(x_lims)
    axs[0].set_ylim(y_lims)

    axs[0].set_title("Ratio of observed to mean stochastic moment release")

    # plot rank
    if map_epsg is None:
        cell_gdf.plot(
            column="moment_over_under_frac",
            ax=axs[1],
            vmin=0.0,
            vmax=1.0,
            cmap="PiYG",
            legend=True,
        )
    else:
        cell_gdf.to_crs(epsg=map_epsg).plot(
            column="moment_over_under_frac",
            ax=axs[1],
            vmin=0.0,
            vmax=1.0,
            cmap="PiYG",
            legend=True,
        )

    world = gpd.read_file(natural_earth_countries_file)
    if map_epsg is None:
        world.plot(ax=axs[1], color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(
            ax=axs[0], color="none", edgecolor="black"
        )
    axs[1].set_xlim(x_lims)
    axs[1].set_ylim(y_lims)

    axs[1].set_title(
        "Rank of observed moment release compared\n"
        + "to stochastic event sets"
    )

    plt.switch_backend("svg")
    fig_str = io.StringIO()
    fig.savefig(fig_str, format="svg")
    fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
    return fig_svg


def plot_rup_match_map(
    eqs, matched_rups, unmatched_eqs=None, map_epsg=None, return_str=False
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    matched_eqs = eqs.loc[matched_rups.index]
    matched_eqs["likelihood"] = matched_rups["likelihood"]

    # Define colormap normalization based on the likelihood range
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"

    if map_epsg is None:
        # Store the output of the plot, which is necessary for colorbar creation
        sc = matched_eqs.plot(
            column="likelihood",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            markersize=matched_eqs.magnitude**3 * 0.1,
        )

        if unmatched_eqs is not None:
            unmatched_eqs.plot(
                ax=ax,
                color="red",
                edgecolor="black",
                markersize=unmatched_eqs.magnitude**3 * 0.1,
            )

    else:
        # When specifying a CRS, plotting still happens in a similar way
        sc = matched_eqs.to_crs(epsg=map_epsg).plot(
            column="likelihood",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            markersize=matched_eqs.magnitude**3 * 0.1,
        )

        if unmatched_eqs is not None:
            unmatched_eqs.to_crs(epsg=map_epsg).plot(
                ax=ax,
                color="red",
                edgecolor="black",
                markersize=unmatched_eqs.magnitude**3 * 0.1,
            )

    # Adjusting the limits might be necessary after plotting the world map
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(natural_earth_countries_file)
    if map_epsg is None:
        world.plot(ax=ax, color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(
            ax=ax, color="none", edgecolor="black"
        )

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Likelihood")

    if return_str:
        plt.switch_backend("svg")
        fig_str = io.StringIO()
        fig.savefig(fig_str, format="svg")
        fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
        return fig_svg
    else:
        return fig


def plot_rup_match_mag_dist(
    matched_rups, eqs, s=6, return_str: bool = True, **kwargs
):
    eq_mags = eqs.loc[matched_rups.index, "magnitude"].values
    rup_mags = matched_rups.magnitude.values
    dists = matched_rups.eq_dist.values
    likes = np.float_(matched_rups.likelihood.values)
    # norm_likes = (likes - likes.min()) / (likes.max() - likes.min())
    colors = plt.cm.viridis(likes)

    lines = [
        [(mag1, dists[i]), (rup_mags[i], dists[i])]
        for i, mag1 in enumerate(eq_mags)
    ]

    fig, ax = plt.subplots(**kwargs)
    ax.scatter(
        rup_mags,
        dists,
        edgecolors=colors,
        facecolors="none",
        label="ruptures",
        s=s,
        lw=0.6,
    )
    ax.scatter(
        eq_mags,
        dists,
        edgecolors=colors,
        facecolors=colors,
        label="earthquakes",
        s=s,
    )

    line_col = LineCollection(lines, colors=colors, linewidths=0.3)
    ax.add_collection(line_col)

    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Match Likelihood")
    ax.legend(loc="best")

    plt.xlabel("Magnitude")
    plt.ylabel("Distance (km)")
    plt.title("Magnitude and Distance for earthquake-rupture matches")

    if return_str:
        plt.switch_backend("svg")
        fig_str = io.StringIO()
        fig.savefig(fig_str, format="svg")
        fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
        return fig_svg

    else:
        return fig
