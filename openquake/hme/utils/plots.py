from typing import Union, Optional, Tuple

import h3
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

import io
from .stats import sample_event_times_in_interval


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

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    if map_epsg is None:
        world.plot(ax=ax, color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(ax=ax, color="none", edgecolor="black")

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
            column="S_bin_pct",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="OrRd_r",
            legend=True,
        )

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    if map_epsg is None:
        world.plot(ax=ax, color="none", edgecolor="black")
    else:
        world.to_crs(epsg=map_epsg).plot(ax=ax, color="none", edgecolor="black")
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.switch_backend("svg")
    fig_str = io.StringIO()
    fig.savefig(fig_str, format="svg")
    fig_svg = "<svg" + fig_str.getvalue().split("<svg")[1]
    return fig_svg


def plot_over_under_map(cell_gdf: GeoDataFrame, map_epsg: Optional[int] = None):
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

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
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

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
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
