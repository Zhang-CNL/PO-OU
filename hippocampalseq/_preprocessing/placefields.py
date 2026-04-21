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
        bin_size_cm: float = 2.0,
        posterior: bool = True
    ) -> np.ndarray:
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

def calculate_linear_placefield(spike_hists, position_hists, axis, pf_gaussian_sd):
    linear_spike_hist = np.sum(spike_hists, axis=axis)
    linear_position_hist = np.sum(position_hists, axis=axis-1)

    nbins = linear_position_hist.shape[0]

    linear_placefield = np.zeros_like(linear_spike_hist)
    for cell in range(linear_spike_hist.shape[0]):
        place_field = linear_spike_hist[cell] / linear_position_hist
        place_field[~np.isfinite(place_field)] = 0

        place_field = gaussian_filter1d(place_field, sigma=pf_gaussian_sd)
        place_field[~np.isfinite(place_field)] = 0
        place_field[place_field < 0] = 0
        linear_placefield[cell] = place_field
    return linear_placefield[...,None]

def calculate_placefields(
        run_position_data,
        run_spike_info,
        run_spikes,
        excitatory_neurons,
        environment_size: Optional[Tuple[int]] = (0,0,200,200),
        track_type = 'Open',
        bin_size_cm: int = 2,
        place_field_gaussian_sd_cm: float = 2.0,
        prior_mean_rat_sps: float = 1.0,
        prior_beta_s: float = .01,
        posterior: bool = True,
        min_spike_rate: float = 1.0,
        velocity_cutoff = 5.0
    ):
    assert track_type in ['Open', 'Linear']
    prior_alpha_s = prior_beta_s * prior_mean_rat_sps + 1
    pf_gaussian_sd = hseu.cm_to_bins(place_field_gaussian_sd_cm, bin_size_cm)

    mask = run_position_data['velocity'].values >= velocity_cutoff
    x  = run_position_data['x'].values[mask]
    y  = run_position_data['y'].values[mask]
    dt = run_position_data['delta t'].values[mask]
    ncells = len(run_spike_info)

    if environment_size is None:
        environment_size = (
            np.min(x),
            np.min(y),
            np.max(x),
            np.max(y)
        )

    spatial_grid_x = np.arange(environment_size[0], environment_size[2], bin_size_cm) + bin_size_cm/2
    spatial_grid_y = np.arange(environment_size[1], environment_size[3], bin_size_cm) + bin_size_cm/2
    nbx = len(spatial_grid_x)
    nby = len(spatial_grid_y)
    print(spatial_grid_x)


    position_hist,_,_ = np.histogram2d(
        x, y,
        bins=(nbx,nby),
        weights=dt
    )
    position_hist = position_hist.T
    print(position_hist.shape)

    spike_hists  = np.zeros((ncells,nby,nbx))
    for cell_id in range(ncells):
        spike_pos = run_spike_info[cell_id]
        cell_v = spike_pos['velocity'].values
        mask   = cell_v >= velocity_cutoff
        cell_x = spike_pos['x'].values[mask]
        cell_y = spike_pos['y'].values[mask]

        if len(cell_x) > 0:
            spike_hist ,_,_ = np.histogram2d(
                cell_x,
                cell_y,
                bins=(nbx,nby),
            )
            spike_hists[cell_id] = spike_hist.T
    if track_type == 'Linear':
        xspan = x.max() - x.min()
        yspan = y.max() - y.min()
        axis = 1 if xspan >= xspan else 2
        place_fields = calculate_linear_placefield(
            spike_hists, 
            position_hist, 
            pf_gaussian_sd=pf_gaussian_sd,
            axis=axis
        )
    else:
        place_fields = np.zeros((ncells, nby,nbx))
        for cell_id in range(ncells):
            place_fields[cell_id] = calculate_one_placefield(
                position_hist,
                spike_hists[cell_id],
                pf_gaussian_sd,
                prior_alpha_s,
                prior_beta_s,
                bin_size_cm,
                posterior
            )

    naxes = place_fields.ndim - 1
    max_firingrate = np.max(place_fields, axis=tuple(range(1,naxes+1)))
    above_thresh = np.squeeze(np.argwhere(max_firingrate > min_spike_rate))
    place_cell_ids = np.intersect1d(excitatory_neurons, above_thresh)

    #return place_fields[place_cell_ids], place_cell_ids
    return place_fields, np.arange(ncells)