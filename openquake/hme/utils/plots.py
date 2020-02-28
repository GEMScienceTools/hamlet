from typing import Union, Optional

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

import io
from .stats import sample_event_times_in_interval


def _sample_n_events(rate):
    return len(sample_event_times_in_interval(rate, interval_length=1.))


def _make_stoch_mfds(mfd, iters: int):

    cum_rates = list(mfd.values())

    incr_rates = []
    for i, rate in enumerate(cum_rates):
        if i < len(cum_rates) - 1:
            incr_rates.append(rate - cum_rates[i + 1])
        elif i == len(cum_rates) - 1:
            incr_rates.append(rate)

    stoch_mfd_vals = []

    for i in range(iters):
        n_event_list = [_sample_n_events(rate) for rate in incr_rates]
        stoch_mfd_vals.append(np.cumsum(n_event_list[::-1])[::-1])

    return stoch_mfd_vals


def plot_mfd(model: Optional[dict] = None,
             model_format: str = 'C0-',
             model_label: str = 'model',
             model_iters: int = 500,
             observed: Optional[dict] = None,
             observed_format: str = 'C3o-.',
             observed_label: str = 'observed',
             return_fig: bool = True,
             return_string: bool = False,
             save_fig: Union[bool, str] = False,
             **kwargs):
    """
    Makes a plot of empirical MFDs

    """
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, yscale='log')
    plt.title('Magnitude-Frequency Distribution')

    if model is not None:
        if model_iters > 0:
            stoch_mfd_vals = _make_stoch_mfds(model, iters=model_iters)
            for smfd in stoch_mfd_vals:
                ax.plot(list(model.keys()),
                        smfd,
                        model_format,
                        lw=10 / model_iters)

    if model is not None:
        ax.plot(list(model.keys()),
                list(model.values()),
                model_format,
                label=model_label)

    if observed is not None:
        ax.plot(list(observed.keys()),
                list(observed.values()),
                observed_format,
                label=observed_label)

    ax.legend(loc='upper right')
    ax.set_ylabel('Annual frequency of exceedance')
    ax.set_xlabel('Magnitude')

    if save_fig is not False:
        fig.savefig(save_fig)

    if return_fig is True:
        return fig

    elif return_string is True:
        plt.switch_backend('svg')
        fig_str = io.StringIO()
        fig.savefig(fig_str, format='svg')
        plt.close(fig)
        fig_svg = '<svg' + fig_str.getvalue().split('<svg')[1]
        return fig_svg


def plot_likelihood_map(bin_gdf: GeoDataFrame,
                        plot_eqs: bool = True,
                        eq_gdf: Optional[GeoDataFrame] = None,
                        map_epsg: Optional[int] = None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if map_epsg is None:
        bin_gdf.plot(column='log_like',
                     ax=ax,
                     vmin=0.,
                     vmax=1.,
                     cmap='OrRd_r',
                     legend=True)
    else:
        bin_gdf.to_crs(epsg=map_epsg).plot(column='log_like',
                                           ax=ax,
                                           vmin=0.,
                                           vmax=1.,
                                           cmap='OrRd_r',
                                           legend=True)

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    if map_epsg is None:
        world.plot(ax=ax, color='none', edgecolor='black')
    else:
        world.to_crs(epsg=map_epsg).plot(ax=ax,
                                         color='none',
                                         edgecolor='black')

    if plot_eqs is True:
        if eq_gdf is not None:
            if map_epsg is None:
                ax.scatter(eq_gdf.longitude,
                           eq_gdf.latitude,
                           s=(eq_gdf.magnitude**3) / 10.,
                           edgecolor='blue',
                           facecolors='none',
                           alpha=0.3)
            else:
                pass

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.switch_backend('svg')
    fig_str = io.StringIO()
    fig.savefig(fig_str, format='svg')
    fig_svg = '<svg' + fig_str.getvalue().split('<svg')[1]
    return fig_svg
