import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from typing import Optional

import hippocampalseq.utils as hseu

def calculate_one_placefield(
        position_hist: np.ndarray,
        spike_hist: np.ndarray,
        place_field_sd_gaussian: float,
        prior_alpha_s: float,
        prior_beta_s: float,
        posterior: bool = True
    ) -> np.ndarray:
    """Calculate a place field from a position histogram and a spike histogram.

    Args:
        position_hist (np.ndarray): Position histogram.
        spike_hist (np.ndarray): Spike histogram.
        place_field_sd_gaussian (float): Standard deviation of the Gaussian filter used to smooth the place field.
        prior_alpha_s (float): Prior alpha parameter for the spike histogram.
        prior_beta_s (float): Prior beta parameter for the position histogram.
        posterior (bool): Whether to use a posterior place field. Defaults to True.

    Returns:
        (np.ndarray): Place field.
    """
    if posterior:
        spike_hist_with_prior = spike_hist + prior_alpha_s - 1
        pos_hist_with_prior_s = position_hist + prior_beta_s
        place_field_raw = spike_hist_with_prior / pos_hist_with_prior_s
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            place_field_raw = np.nan_to_num(spike_hist / position_hist, posinf=0, neginf=0)
    place_field_smoothed = gaussian_filter(
        place_field_raw, sigma=place_field_sd_gaussian
    )
    return place_field_smoothed

def calculate_one_linear_placefield(
        linear_spike_hist: np.ndarray,
        linear_position_hist: np.ndarray,
        pf_gaussian_sd: float,
        prior_alpha_s: float,
        prior_beta_s: float,
        posterior: bool = True
    ) -> np.ndarray:
    """Calculate a linear place field from a position histogram and a spike histogram.
    In the linear case it sums along either the x or y axis depending on which is chosen.

    Args:
        position_hist (np.ndarray): Position histogram.
        spike_hist (np.ndarray): Spike histogram.
        pf_sd_gaussian (float): Standard deviation of the Gaussian filter used to smooth the place field.
        prior_alpha_s (float): Prior alpha parameter for the spike histogram.
        prior_beta_s (float): Prior beta parameter for the position histogram.
        posterior (bool): Whether to use a posterior place field. Defaults to True.
    Returns:
        (np.ndarray): Linear place field.
    """

    if posterior:
        linear_spike_hist_with_prior = linear_spike_hist + prior_alpha_s - 1
        linear_position_hist_with_prior_s = linear_position_hist + prior_beta_s
        place_field = linear_spike_hist_with_prior / linear_position_hist_with_prior_s
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            place_field = linear_spike_hist[cell] / linear_position_hist
            place_field[~np.isfinite(place_field)] = 0

    place_field = gaussian_filter1d(place_field, sigma=pf_gaussian_sd)
    #place_field[~np.isfinite(place_field)] = 0
    place_field[place_field < 0] = 0
    return place_field

def calculate_placefields(
        run_position_data: nap.TsdFrame,
        run_spike_info: dict[int, nap.TsdFrame],
        excitatory_neurons: np.ndarray,
        environment_size: Optional[list[tuple[int,...]]] = [(0,200),(0,200)],
        track_type = 'Linear',
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 4.0,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        posterior: bool = True,
        min_spike_rate: float = 1.0,
        velocity_cutoff: float = 5.0,
        flatten_linear: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int,...]]]:
    """Calculate place fields from position and spike data.
    We want to use the 'ij' indexing convention for the place field data, meaning that
    if we want to plot the place field using imshow, we must transpose it first.
    In addition, given a set of points (x,y), indexing of a grid is performed like so:
    grid[x,y].

    Args:
        run_position_data (nap.TsdFrame): X,Y, and delta time data.
        run_spike_info (dict[int, nap.TsdFrame]): Position data corresponding to, and aligned with, individual cells.
        excitatory_neurons (np.ndarray): Excitatory neuron ids.
        environment_size (Optional[list[tuple[int,...]]): Environment size. If None, adapts bins from data. Defaults to (0,0,200,200).
        track_type (str): Track type. Can be one of ("Open", "Linear"). Defaults to 'Linear'.
        bin_size_cm (int): Bin size in centimeters. Defaults to 2.
        place_field_gaussian_sd_cm (float): Standard deviation of the Gaussian filter used to smooth the place field in centimeters. Defaults to 4.0.
        prior_mean_rat_sps (float): Prior mean ratio of spikes per second. Defaults to 1.0.
        prior_beta_s (float): Prior beta parameter for the position histogram. Defaults to .01.
        posterior (bool): Whether to use a posterior place field. Defaults to True.
        min_spike_rate (float): Minimum spike rate. Defaults to 1.0.
        velocity_cutoff (float): Velocity cutoff in cm/s. Defaults to 5.0.
        flatten_linear (bool): Flatten linear place fields to a single dimension corresponding to the longer span. Defaults to False.

    Returns:
        (np.ndarray): Place fields.
        (np.ndarray): Place cell ids.
        (np.ndarray): Number of times the rat was in a specific position.
        (list[tuple[int,...]]): Size of the environment. Useful if we learn it from the data.
    """
    assert track_type in ['Open', 'Linear']
    prior_alpha_s = prior_beta_s * prior_mean_rat_sps + 1
    pf_gaussian_sd = hseu.cm_to_bins(place_field_gaussian_sd_cm, bin_size_cm)

    mask = run_position_data['Velocity'].values >= velocity_cutoff
    x  = run_position_data['x'].values[mask]
    y  = run_position_data['y'].values[mask]
    dt = run_position_data['Delta t'].values[mask]
    ncells = len(run_spike_info)

    if environment_size is None:
        environment_size = [
            (np.min(x),np.max(x)),
            (np.min(y),np.max(y))
        ]

    xbounds,ybounds = environment_size
    nbx = int((xbounds[1] - xbounds[0]) / bin_size_cm)
    nby = int((ybounds[1] - ybounds[0]) / bin_size_cm)
    spatial_grid_x = np.linspace(xbounds[0], xbounds[1], nbx + 1) + bin_size_cm / 2
    spatial_grid_y = np.linspace(ybounds[0], ybounds[1], nby + 1) + bin_size_cm / 2

    position_hist,xedges,yedges = np.histogram2d(
        x, y,
        bins=(spatial_grid_x,spatial_grid_y),
        weights=dt
    )

    spike_hists  = np.zeros((ncells,nbx,nby))
    for cell_id in range(ncells):
        spike_pos = run_spike_info[cell_id]
        cell_v = spike_pos['Velocity'].values
        mask   = cell_v >= velocity_cutoff
        cell_x = spike_pos['x'].values[mask]
        cell_y = spike_pos['y'].values[mask]

        if len(cell_x) > 0:
            spike_hist,_,_ = np.histogram2d(
                cell_x,
                cell_y,
                bins=(spatial_grid_x,spatial_grid_y),
            )
            spike_hists[cell_id] = spike_hist

    if flatten_linear and track_type == 'Linear':
        xspan = x.max() - x.min()
        yspan = y.max() - y.min()
        if xspan >= yspan:
            axis = 1
            environment_size = [
                (0, (len(xedges) - 1) * bin_size_cm),
            ]
        else:
            axis = 2
            environment_size = [
                (0, (len(yedges) - 1) * bin_size_cm)
            ]

        linear_spike_hist = np.sum(spike_hists, axis=axis)
        position_hist = np.sum(position_hist, axis=axis-1)
        place_fields = np.zeros_like(linear_spike_hist)
        for cell_id in range(ncells):
            place_fields[cell_id] = calculate_one_linear_placefield(
                linear_spike_hist,
                position_hist,
                pf_gaussian_sd=pf_gaussian_sd,
            )
        place_fields = place_fields[...,None]
    else:
        if track_type == 'Linear':
            environment_size = [
                (0, (len(xedges) - 1) * bin_size_cm),
                (0, (len(yedges) - 1) * bin_size_cm),
            ]
        place_fields = np.zeros((ncells, nbx, nby))
        for cell_id in range(ncells):
            place_fields[cell_id] = calculate_one_placefield(
                position_hist,
                spike_hists[cell_id],
                pf_gaussian_sd,
                prior_alpha_s,
                prior_beta_s,
                posterior
            )

    # Filter out place fields that are from inhibitory neurons
    # or fall below the minimum firing rate threshold.
    naxes = place_fields.ndim - 1
    max_firingrate = np.max(place_fields, axis=tuple(range(1,naxes+1)))
    above_thresh = np.squeeze(np.argwhere(max_firingrate > min_spike_rate))
    place_cell_ids = np.intersect1d(excitatory_neurons, above_thresh)

    return (
        place_fields, 
        place_cell_ids, 
        position_hist,
        environment_size
    )
