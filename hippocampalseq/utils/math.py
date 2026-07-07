import numpy as np
import numpy.typing as npt
import torch 
from typing import Tuple

from .utils import changeover_functions

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

    The log of our poission distribution becomes $x_{t,i}\cdot ln(f_i(z_t)\gamma\delta t) - f_i(z_t)\gamma\delta t - ln(x_{t,i}!)$

    Args:
        spikemat (npt.ArrayLike): Spikemat of shape (T, Ncell) $x_{t,i,j}$
        place_fields (npt.ArrayLike): Place fields of shape (Ncell, Nbx, Nby) $f_{i,j}(z_t)$
        dt (float|torch.Tensor): Time window in seconds

    Returns:
        (npt.ArrayLike): (T, Nbx, Nby) matrix of emission probabilities
    """
    sum,log,einsum,amax = changeover_functions(type(spikemat), 'sum', 'log', 'einsum', 'amax')
    lambdas = place_fields * dt
    
    sum_lambda = sum(lambdas, axis=0)
    
    log_lambdas = log(lambdas + 1e-10)
    term1 = einsum('tn,nhw->thw', spikemat, log_lambdas)
    log_likelihood_maps = term1 - sum_lambda
    
    # Numerical stability trick per time bin
    # Subtract max along spatial dimensions (H, W) for each T
    max_log = amax(log_likelihood_maps, axis=(1, 2), keepdims=True)
    
    return log_likelihood_maps - max_log

def calc_poisson_emission_probabilities_2d(
        spikemat: npt.ArrayLike,
        place_fields: npt.ArrayLike,
        dt: float|npt.ArrayLike
    ) -> npt.ArrayLike:
    exp = changeover_functions(type(spikemat), 'exp')
    log_emission = calc_poisson_emission_probabilities_log_2d(spikemat, place_fields, dt)
    return exp(log_emission)

def analytical_gaussian_approximation(
        z: torch.Tensor, 
        pz: torch.Tensor, 
        bin_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Nx, Ny = pz.shape
    if z.ndim == 2:
        z = z.unsqueeze(0).expand(B,-1,-1) 

    D = z.shape[-1]
    
    mu = torch.sum(pz.reshape(B, Nx*Ny, 1) * z, dim=1) / torch.sum(pz,dim=(1,2))[:,None]
    
    z_centered = z - mu.unsqueeze(1) # (B, N, D)
    z_centered = z_centered.unsqueeze(-1)
    
    outer_products = z_centered @ z_centered.transpose(-2,-1)
    sigma = torch.sum(pz.view(B, Nx*Ny, 1, 1) * outer_products, dim=1) # (B, D, D)
    
    return mu.unsqueeze(-1), sigma

def mT(x: np.ndarray|torch.Tensor):
    """Shorthand for np.matrix_transpose or torch.Tensor.mT"""
    if isinstance(x, torch.Tensor):
        return x.mT
    return np.matrix_transpose(x)

def invmul(A: np.ndarray|torch.Tensor, B: np.ndarray|torch.Tensor):
    """Computes :math:`AB^{-1}`"""
    if isinstance(A, torch.Tensor):
        return mT(torch.linalg.solve(mT(B), mT(A)))
    return mT(np.linalg.solve(mT(B), mT(A))) # Equivalent to A @ np.linalg.inv(B)

def mulinv(B: np.ndarray|torch.Tensor, A: np.ndarray|torch.Tensor):
    """Computes :math:`B^{-1}A`"""
    if isinstance(A, torch.Tensor):
        return torch.linalg.solve(B, A)
    return np.linalg.solve(B, A)
