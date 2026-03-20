import mat73 
import scipy.io as sio
from scipy.special import factorial
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from itertools import product
import numpy as np
import torch

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


def read_mat(file: str):
    try:
        return mat73.loadmat(file)
    except:
        return sio.loadmat(file, squeeze_me=True, struct_as_record=False)

def extract_times_from_boolean(boolean_arr, run_times):
    start_times = []
    end_times = []
    previous_val = boolean_arr[0]
    if previous_val:
        start_times.append(run_times[0])
    for count, val in enumerate(boolean_arr[1:]):
        i = count + 1
        if val != previous_val:
            if val:
                start_times.append(run_times[i])
            else:
                end_times.append(run_times[i])
        previous_val = val
    if val:
        end_times.append(run_times[-1])
    return np.array(start_times), np.array(end_times)

def times_to_bool(data_times, start_time, end_time):
    times_after_start = data_times >= start_time
    times_before_end = data_times <= end_time
    window_ind = times_after_start & times_before_end
    return window_ind

def cm_to_bins(array_in_cm, bin_size_cm: int = 4):
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
    return tuning_curves.reshape((len(place_cell_ids), np.prod(tuning_curves.shape[1:3])))

# Poisson distribution:  $Pois(k=x_{t,i}|\lambda=f_i(z_t)\gamma\delta t) = \frac{\lambda^ke^{-\lambda}}{k!}$

# Log-version becomes: $k\cdot ln(\lambda) -\lambda - ln(k!)$
# $x_{t,i}\cdot ln(f_i(z_t)\gamma\delta t) - f_i(z_t)\gamma\delta t - ln(x_{t,i}!)$

def calc_poisson_emission_probabilities_log(
        spikemat     : np.ndarray,
        place_fields : np.ndarray, 
        dt: float
    ) -> np.ndarray:
    r"""This function calculates emission probabilities $ln\ P(x_t|z_t) = ln\ \prod_{i=1}^N Pois(x_{t,i}|f_i(z_t)\gamma\delta t)$
    Args:
        spikemat (np.ndarray): Spikemat of shape (T, Ncells) $x_{t,i}$
        place_fields (np.ndarray): Place fields of shape (Ncells, Ngrid) $f_i(z_t)$
        dt(float): Time window in seconds
    Returns:
        (np.ndarray): (Ngrid, T) matrix of emission probabilities
    """
    sum, log = changeover_functions(type(spikemat), 'sum', 'log')
    fac = factorial if type(spikemat) == np.ndarray else lambda x: torch.exp(torch.lgamma(x+1))

    log_pfs = log(place_fields)#.T

    # $\sum_n x_t ln(f_n(z_t))$
    pf_spikes_sum = spikemat @ log_pfs# (log_pfs @ spikemat.T).T # (T,Ngrid) matrix

    #  $\sum_n x_t ln(\delta_t)$
    time_window_spikes_sum = sum(spikemat * log(dt), axis=1) # (T,) matrix

    # $\delta t\sum_n f_n(z_t)$
    pf_sum = dt * sum(place_fields, axis=0) # (Ngrid,) matrix

    # $\sum_n ln(x_{t,n}!)$
    spikes_sum = sum(log(fac(spikemat)), axis=1) # (T,) matrix
    # calculate emission probs
    pf_sum_norm = (pf_spikes_sum.T + time_window_spikes_sum).T - pf_sum
    emission_probabilities_log = pf_sum_norm.T - spikes_sum
    return emission_probabilities_log


def calc_poisson_emission_probabilities(
        spikemat     : np.ndarray, 
        place_fields : np.ndarray, 
        dt: float
    ) -> np.ndarray:
    r"""
    Calculate emission probabilities $P(x_t|z_t) = \prod_{i=1}^N Pois(x_{t,i}|f_i(z_t)\gamma\delta t)$

    Args:
        spikemat (np.ndarray): Spikemat of shape (T, Ncells) $x_{t,i}$
        place_fields (np.ndarray): Place fields of shape (Ncells, Ngrid) $f_i(z_t)$
        dt (float): Time window in seconds

    Returns:
        (np.ndarray): (Ngrid, T) matrix of emission probabilities
    """
    exp = changeover_functions(type(spikemat), 'exp')
    emission_probabilities_log = calc_poisson_emission_probabilities_log(
        spikemat, place_fields, dt
    )
    emission_probabilities = exp(emission_probabilities_log)
    return emission_probabilities

def calc_poisson_emission_probabilities_log_2d(
        spikemat     : np.ndarray|torch.Tensor,
        place_fields : np.ndarray|torch.Tensor, 
        dt: float|torch.Tensor,
    ) -> np.ndarray|torch.Tensor:
    r"""Calculate emission probabilities $ln\ P(x_t|z_t) = ln\ \prod_{i,j} Pois(x_{t,i,j}f_{i,j}(z_t)\gamma\delta t)$ for a 2D place field.
    Same function as `calc_poisson_emission_probabilities_log` except the output is a 2D matrix.

    Args:
        spikemat (np.ndarray|torch.Tensor): Spikemat of shape (T, Ncell) $x_{t,i,j}$
        place_fields (np.ndarray|torch.Tensor): Place fields of shape (Ncell, Nbx, Nby) $f_{i,j}(z_t)$
        dt (float|torch.Tensor): Time window in seconds

    Returns:
        (np.ndarray|torch.Tensor): (T, Nbx, Nby) matrix of emission probabilities
    """
    log, sum, einsum = changeover_functions(type(spikemat), 'log', 'sum', 'einsum')
    fac = factorial if type(spikemat) == np.ndarray else lambda x: torch.exp(torch.lgamma(x+1))

    log_pfs = log(place_fields)

    #pf_spikes_sum = spikemat[None,:,:] @ log_pfs # (T,n_bx,n_by) matrix
    pf_spikes_sum = einsum('ij,jkl->ikl', spikemat, log_pfs)

    time_window_spikes_sum = sum(spikemat * log(dt), axis=1) # (T,) matrix

    pf_sum = dt * sum(place_fields, axis=0) # (n_bx,n_by) matrix

    spikes_sum = sum(log(fac(spikemat)), axis=1) # (T,) matrix
    # calculate emission probs
    pf_sum_norm = (pf_spikes_sum + time_window_spikes_sum[:,None,None]) - pf_sum
    emission_probabilities_log = pf_sum_norm - spikes_sum[:,None,None]
    return emission_probabilities_log

def calc_poisson_emission_probabilities_2d(
        spikemat: np.ndarray|torch.Tensor,
        place_fields: np.ndarray|torch.Tensor,
        dt: float|torch.Tensor
    ) -> np.ndarray|torch.Tensor:
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
    spikemat = np.empty(shape=(0, len(place_cell_ids)), dtype=int)
    timebin_start_time = start_time
    timebin_end_time = start_time + time_window_s
    while timebin_end_time < end_time:
        spikes_after_start = spike_times >= timebin_start_time
        spikes_before_end = spike_times < timebin_end_time
        timebin_bool = spikes_after_start == spikes_before_end
        spike_ids_in_window = spike_ids[timebin_bool]
        spikevector = np.array(
            [[np.sum(spike_ids_in_window == cell_id) for cell_id in place_cell_ids]]
        )
        spikemat = np.append(spikemat, spikevector, axis=0)
        timebin_start_time = timebin_start_time + time_window_advance_s
        timebin_end_time = timebin_end_time + time_window_advance_s
    return np.array(spikemat, dtype=int)

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
