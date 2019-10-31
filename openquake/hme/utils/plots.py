import io
from typing import Union, Optional

import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import GeoDataFrame


def plot_mfd(model: Optional[dict] = None,
             model_format: str = 'C0-',
             model_label: str = 'model',
             observed: Optional[dict] = None,
             observed_format: str = 'C1o-.',
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
                        eq_gdf: Optional[GeoDataFrame] = None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    bin_gdf.plot(column='log_like',
                 ax=ax,
                 vmin=0.,
                 vmax=1.,
                 cmap='OrRd_r',
                 legend=True)

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='none', edgecolor='black')

    if eq_gdf is not None:
        ax.scatter(eq_gdf.longitude,
                   eq_gdf.latitude,
                   s=(eq_gdf.magnitude**3) / 10.,
                   edgecolor='blue',
                   facecolors='none')

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.switch_backend('svg')
    fig_str = io.StringIO()
    fig.savefig(fig_str, format='svg')
    fig_svg = '<svg' + fig_str.getvalue().split('<svg')[1]
    return fig_svg