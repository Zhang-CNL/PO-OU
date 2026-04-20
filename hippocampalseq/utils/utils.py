import os
import mat73 
import scipy.io as sio
from scipy.special import factorial
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from itertools import product
import numpy as np
import numpy.typing as npt
import torch
import compress_pickle

class AttrDict(dict):
    def __init__(self, dct):
        super().__init__(dct)
        self.__dict__ = dct

    def __setitem__(self, k, v):
        super().__setitem__(self, k, v)
        self.__dict__[k] = v

    def __copy__(self):
        return AttrDict(self)

def changeover_functions(type, *args):
    module = torch if type == torch.Tensor else np
    attrs = []
    for arg in args:
        attrs.append(getattr(module, arg))
    if len(attrs) == 1:
        return attrs[0]
    return attrs

def save_pickle(data, fname):
    s = compress_pickle.dumps(data, "gzip")
    with open(fname, 'wb') as f:
        f.write(s)

def read_pickle(fname):
    with open(fname, 'rb') as f:
        raw = f.read()
    return compress_pickle.loads(raw, "gzip")

def read_mat(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found, make sure you have the complete dataset.")
    try:
        return mat73.loadmat(file)
    except:
        return sio.loadmat(file, squeeze_me=True, struct_as_record=False)

def extract_times_from_boolean(boolean_arr, run_times):
    start_times = []
    end_times   = []
    prev        = boolean_arr[0]
    if prev:
        start_times.append(run_times[0])
    for count, val in enumerate(boolean_arr[1:]):
        i = count + 1
        if val != prev:
            if val:
                start_times.append(run_times[i])
            else:
                end_times.append(run_times[i])
        prev = val
    if val:
        end_times.append(run_times[-1])
    return np.array(start_times), np.array(end_times)

def restrict_indices(t_array, start, end):
    start_ind = np.searchsorted(t_array, start, side='left')
    end_ind   = np.searchsorted(t_array, end, side='right')
    return slice(start_ind, end_ind)

def times_to_bool(data_times, start_time, end_time):
    times_after_start = data_times >= start_time
    times_before_end = data_times <= end_time
    window_ind = times_after_start & times_before_end
    return window_ind

def cm_to_bins(array_in_cm, bin_size_cm: int = 2):
    return np.floor(array_in_cm / bin_size_cm)  # cm to bins

def placefield_matrix(place_fields: np.ndarray, place_cell_ids: np.ndarray) -> np.ndarray:
    """Convert a 3d array of place fields to a 2d array, where each row corresponds to a cell id.

    Args:
        place_fields (np.ndarray): 3d array of place fields, where each row is a cell id and each column is a place field.
        place_cell_ids (np.ndarray): 1d array of cell ids.

    Returns:
        (np.ndarray): 2d array of place fields, where each row is a cell id and each column is a place field.
    """
    tuning_curves = place_fields[place_cell_ids]
    return tuning_curves.reshape((len(place_cell_ids), -1))

# Poisson distribution:  $Pois(k=x_{t,i}|\lambda=f_i(z_t)\gamma\delta t) = \frac{\lambda^ke^{-\lambda}}{k!}$

# Log-version becomes: $k\cdot ln(\lambda) -\lambda - ln(k!)$
# $x_{t,i}\cdot ln(f_i(z_t)\gamma\delta t) - f_i(z_t)\gamma\delta t - ln(x_{t,i}!)$

def calc_poisson_emission_probabilities_log_2d(
        spikemat: npt.ArrayLike,
        place_fields: npt.ArrayLike,
        dt: float|npt.ArrayLike
    ) -> npt.ArrayLike:
    r"""Calculate emission probabilities $ln\ P(x_t|z_t) = ln\ \prod_{i,j} Pois(x_{t,i,j}f_{i,j}(z_t)\gamma\delta t)$ for a 2D place field.
    Same function as `calc_poisson_emission_probabilities_log` except the output is a 2D matrix.

    Args:
        spikemat (npt.ArrayLike): Spikemat of shape (T, Ncell) $x_{t,i,j}$
        place_fields (npt.ArrayLike): Place fields of shape (Ncell, Nbx, Nby) $f_{i,j}(z_t)$
        dt (float|torch.Tensor): Time window in seconds

    Returns:
        (npt.ArrayLike): (T, Nbx, Nby) matrix of emission probabilities
    """
    sum,log,einsum,max = changeover_functions(type(spikemat), 'sum', 'log', 'einsum', 'max')
    lambdas = place_fields * dt
    
    sum_lambda = sum(lambdas, axis=0)
    
    log_lambdas = log(lambdas + 1e-10)
    term1 = einsum('tn,nhw->thw', spikemat, log_lambdas)
    log_likelihood_maps = term1 - sum_lambda
    
    # Numerical stability trick per time bin
    # Subtract max along spatial dimensions (H, W) for each T
    max_log = max(log_likelihood_maps, axis=(1, 2), keepdims=True)
    
    return log_likelihood_maps - max_log

def calc_poisson_emission_probabilities_2d(
        spikemat: npt.ArrayLike,
        place_fields: npt.ArrayLike,
        dt: float|npt.ArrayLike
    ) -> npt.ArrayLike:
    exp = changeover_functions(type(spikemat), 'exp')
    lemission = calc_poisson_emission_probabilities_log_2d(spikemat, place_fields, dt)
    emission_probabilities = exp(lemission)
    return emission_probabilities

def extract_spikemat(
        spike_ids: np.ndarray, 
        spike_times: np.ndarray,
        place_cell_ids: np.ndarray,
        start_time: float, 
        end_time: float,
        time_window_s: float,
        time_window_advance_s: float
    ) -> np.ndarray:
    """
    Extracts a spikemat (a matrix where each row corresponds
     to a timebin and each column corresponds to a place cell) from the given spike data.

    Args:
        spike_ids (np.ndarray): IDs of the spikes
        spike_times (np.ndarray): Times of the spikes
        place_cell_ids (np.ndarray): IDs of the place cells
        start_time (float): Start time of the epoch
        end_time (float): End time of the epoch
        time_window_s (float): Length of time window in seconds
        time_window_advance_s (float): Advance of time window in seconds

    Returns:
        np.ndarray: An individual spike matrix.
    """
    sid = np.searchsorted(spike_times, start_time, side='left')
    eid = np.searchsorted(spike_times, end_time, side='right')
    start = spike_times[sid]
    end   = spike_times[eid]
    bin_starts = np.arange(start, end, time_window_advance_s)
    #bin_starts = np.arange(start_time, end_time - time_window_s, time_window_advance_s)

    
    if np.isclose(time_window_s, time_window_advance_s):
        place_idx = np.isin(spike_ids, place_cell_ids)
        s_ids     = spike_ids[place_idx]
        s_times   = spike_times[place_idx]

        time_edges = np.append(bin_starts, bin_starts[-1] + time_window_s)
        cell_edges = np.append(place_cell_ids, place_cell_ids[-1])
        spikemat, _, _ = np.histogram2d(s_times, s_ids, bins=[time_edges, cell_edges])
        return spikemat.astype(int)
    else:
        num_bins = len(bin_starts)
        num_cells = len(place_cell_ids)
        spikemat = np.zeros((num_bins, num_cells), dtype=int)
        
        s_times = spike_times
        s_ids = spike_ids
        u_ids = np.unique(s_ids)
        
        for i, start in enumerate(bin_starts):
            end = start + time_window_s
            idx_start = np.searchsorted(s_times, start, side='left')
            idx_end   = np.searchsorted(s_times, end, side='left')
            
            window_spike_ids = s_ids[idx_start:idx_end]
            
            if len(window_spike_ids) > 0:
                counts = np.bincount(window_spike_ids, minlength=int(u_ids.max()+1))
                spikemat[i, :] = counts[place_cell_ids]
                
        return spikemat

def bin_points(x,y):
    vstack,hstack = changeover_functions(type(x), 'vstack', 'hstack')
    grid = vstack((x,y))
    prod = product(*[grid[dim,:] for dim in range(2)])
    z = vstack([hstack(p) for p in prod])
    return z

def construct_test_mvn(n_points, dz, mu, sigma):
    """Construct test multivariate normal distribution centered around mu

    Args:
        n_points (int): Number of points in each dimension for the test distribution.
        dz (float): Grid spacing in each dimension.
        mu (list): Mean of the test distribution.
        sigma (list): Covariance of the test distribution.

    Returns:
        (np.ndarray): Grid points for the test distribution.
        (np.ndarray): Probability values for the test distribution.
    """
    n_dims = len(mu)
    z = np.meshgrid(*[np.arange(-n_points//2 + mu[i], n_points//2 + mu[i], dz) for i in range(n_dims)])
    #z = np.meshgrid(*[np.arange(-n_points//2 , n_points//2 , dz) for i in range(n_dims)])
    #print("Reloaded")
    z = np.column_stack([_z.ravel() for _z in z])

    rv = multivariate_normal(mean=mu, cov=sigma)
    #pz = np.array([rv.pdf(_z) for _z in z])
    pz = rv.pdf(z)
    return np.array(z),pz

def atleast_2d(x):
    """Ensure that the input array has at least 2 dimensions.
    Differs from np.atleast_2d in that it appends the dimension instead of prepending it

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Input array with at least 2 dimensions.
    """
    if x.ndim < 2:
        x = x[:,None]
    return x

def mT(x):
    """Shorthand for np.matrix_transpose"""
    if type(x) == torch.Tensor:
        return x.mT
    return np.matrix_transpose(x)

def invmul(A, B):
    """Computes :math:`AB^{-1}`"""
    if type(A) == torch.Tensor:
        return mT(torch.linalg.solve(mT(B), mT(A)))
    return mT(np.linalg.solve(mT(B), mT(A))) # Equivalent to A @ np.linalg.inv(B)

def mulinv(B, A):
    """Computes :math:`B^{-1}A`"""
    if type(A) == torch.Tensor:
        return torch.linalg.solve(B, A)
    return np.linalg.solve(B, A)
