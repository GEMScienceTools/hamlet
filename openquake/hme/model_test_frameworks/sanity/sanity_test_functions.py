import logging
from multiprocessing import Pool

from openquake.hme.utils.utils import get_cell_eqs, get_cell_rups, _n_procs


def max_check_function(
    input_data,
    bin_width,
    warn: bool = True,
    parallel: bool = True,
) -> dict:

    cell_ids = list(input_data["cell_groups"].groups.keys())

    if parallel:
        args = (
            (
                get_cell_rups(
                    cell_id,
                    input_data["rupture_gdf"],
                    input_data["cell_groups"],
                ),
                get_cell_eqs(
                    cell_id, input_data["eq_gdf"], input_data["eq_groups"]
                ),
                bin_width,
            )
            for cell_id in cell_ids
        )

        with Pool(_n_procs) as p:
            max_check_results = p.map(check_cell_max_mag_inner_par, args)

        max_check_results_dict = {
            cell_id: max_check_results[i] for i, cell_id in enumerate(cell_ids)
        }

    else:
        max_check_results_dict = {
            cell_id: get_cell_info_and_check_max_mag(
                input_data, cell_id, bin_width
            )
            for cell_id in cell_ids
        }

    if warn:
        for cell, result in max_check_results_dict.items():
            if result is False:
                logging.warn("bin {} fails max mag test.".format(cell))

    return max_check_results_dict


def get_cell_info_and_check_max_mag(input_data, cell_id, bin_width):

    cell_rup_df = get_cell_rups(
        cell_id, input_data["rupture_gdf"], input_data["cell_groups"]
    )
    cell_eq_df = get_cell_eqs(
        cell_id, input_data["eq_gdf"], input_data["eq_groups"]
    )

    return check_cell_max_mag_inner(cell_rup_df, cell_eq_df, bin_width)


def check_cell_max_mag_inner_par(args):
    return check_cell_max_mag_inner(*args)


def check_cell_max_mag_inner(cell_rups, cell_eqs, bin_width):
    # True passes the test, False fails
    return cell_rups.magnitude.max() + bin_width / 2 > cell_eqs.magnitude.max()
