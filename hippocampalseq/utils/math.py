import numpy as np
import torch
import scipy
from typing import Callable,Any,Optional
from .utils import changeover_functions,NDArray

def optimize(
        closure: Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor],
        parameters: list[torch.Tensor],
        closure_kwargs: dict[str, Any],
        autograd_kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Optimize a given set of variables using pytorch's autograd feature.

    Args:
        closure (Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor]): A function used to compute the log-likelihood.
        parameters (list[torch.Tensor]): The parameters to optimize. Passed as the first argument to `closure()`.
        closure_kwargs (dict[str, Any]): Keyword arguments for the loss function. Passed as the second argument to `closure()`.
        autograd_kwargs (dict[str, Any]): Keyword arguments for the optimizer.

    Returns:
        torch.Tensor: The final negative log likelihood.
        List[torch.Tensor]: The optimized parameters.
    """
    for p in parameters:
        p.requires_grad_(True)

    optimizers = {
        'Adam'  : torch.optim.Adam,
        'SGD'   : torch.optim.SGD,
        'AdamW' : torch.optim.AdamW,
        'LBFGS' : torch.optim.LBFGS
    }
    optimizer = optimizers[autograd_kwargs.get('optimizer', 'Adam')](
        parameters,
        lr=autograd_kwargs.get('lr', .01)
    )

    prev_loss = np.inf

    def wrapped_closure():
        optimizer.zero_grad()
        loss = closure(parameters, closure_kwargs)
        loss.backward()
        return loss

    for epoch in range(autograd_kwargs.get('n_epochs', 1000)):
        loss = optimizer.step(wrapped_closure)
        if epoch > 0 and abs((loss.item() - prev_loss) / prev_loss) < autograd_kwargs.get('gd_tol', 1e-3):
            break
        prev_loss = loss.item()

    return loss.detach(),parameters


# Poisson distribution:  $Pois(k=x_{t,i}|\lambda=f_i(z_t)\gamma\delta t) = \frac{\lambda^ke^{-\lambda}}{k!}$

# Log-version becomes: $k\cdot ln(\lambda) -\lambda - ln(k!)$
# $x_{t,i}\cdot ln(f_i(z_t)\gamma\delta t) - f_i(z_t)\gamma\delta t - ln(x_{t,i}!)$

def calc_poisson_emission_probabilities_log_2d(
        spikemat: NDArray,
        place_fields: NDArray,
        dt: float|NDArray
    ) -> NDArray:
    r"""Calculate emission probabilities $ln\ P(x_t|z_t) = ln\ \prod_{i,j} Pois(x_{t,i,j}f_{i,j}(z_t)\gamma\delta t)$ for a 2D place field.
    Same function as `calc_poisson_emission_probabilities_log` except the output is a 2D matrix.

    The log of our poission distribution becomes $x_{t,i}\cdot ln(f_i(z_t)\gamma\delta t) - f_i(z_t)\gamma\delta t - ln(x_{t,i}!)$

    Args:
        spikemat (NDArray): Spikemat of shape (T, Ncell) $x_{t,i,j}$
        place_fields (NDArray): Place fields of shape (Ncell, Nbx, Nby) $f_{i,j}(z_t)$
        dt (float|torch.Tensor): Time window in seconds

    Returns:
        (NDArray): (T, Nbx, Nby) matrix of emission probabilities
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
        spikemat: NDArray,
        place_fields: NDArray,
        dt: float|NDArray
    ) -> NDArray:
    exp = changeover_functions(type(spikemat), 'exp')
    log_emission = calc_poisson_emission_probabilities_log_2d(spikemat, place_fields, dt)
    return exp(log_emission)

def analytical_gaussian_approximation(
        z: torch.Tensor,
        pz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Nx, Ny = pz.shape
    if z.ndim == 2:
        z = z.unsqueeze(0).expand(B,-1,-1)

    mu = torch.sum(pz.reshape(B, Nx*Ny, 1) * z, dim=1) / torch.sum(pz,dim=(1,2))[:,None]

    z_centered = z - mu.unsqueeze(1) # (B, N, D)
    z_centered = z_centered.unsqueeze(-1)

    outer_products = z_centered @ z_centered.transpose(-2,-1)
    sigma = torch.sum(pz.view(B, Nx*Ny, 1, 1) * outer_products, dim=1) # (B, D, D)

    return mu.unsqueeze(-1), sigma

def mT(x: NDArray) -> NDArray:
    """Shorthand for np.matrix_transpose or torch.Tensor.mT"""
    if isinstance(x, torch.Tensor):
        return x.mT
    return np.matrix_transpose(x)

def invmul(A: NDArray, B: NDArray) -> NDArray:
    """Computes :math:`AB^{-1}`"""
    if isinstance(A, torch.Tensor):
        return mT(torch.linalg.solve(mT(B), mT(A)))
    return mT(np.linalg.solve(mT(B), mT(A))) # Equivalent to A @ np.linalg.inv(B)

def mulinv(B: NDArray, A: NDArray) -> NDArray:
    """Computes :math:`B^{-1}A`"""
    if isinstance(A, torch.Tensor):
        return torch.linalg.solve(B, A)
    return np.linalg.solve(B, A)

def orthog(X: NDArray) -> NDArray:
    if isinstance(X, torch.Tensor):
        U,S,V = torch.linalg.svd(X)
        M,N = U.shape[0],V.shape[1]
        rcond = np.finfo(S.dtype).eps * max(M,N)
        tol = np.amax(S, initial=0.) * rcond
        num = np.sum(S > tol, dtype=int)
        return U[:,:num]
    else:
        return scipy.linalg.orth(X)

def pca(X: NDArray, n_components: Optional[int] = None) -> tuple[NDArray,...]:
    mean,cov,argsort,dot = changeover_functions(type(X), 'mean', 'cov', 'argsort', 'dot')

    centered_x = X - mean(X, axis=0)
    _cov = cov(centered_x.T)
    if isinstance(X, torch.Tensor):
        eval,evec = torch.linalg.eigh(_cov)
    else:
        eval,evec = np.linalg.eigh(_cov)

    idx = argsort(eval)[::-1]
    eval = eval[idx]
    evec = evec[:,idx]
    if n_components is not None:
        evec = evec[:,:n_components]
        eval = eval[:n_components]

    return (
        dot(centered_x, evec),
        eval,
        evec
    )

def find_halfheight_peaks(signal: np.ndarray, peak_idx: int) -> np.ndarray:
    """Reimplementation of MATLAB's findpeaks with WidthReference='halfheight'.
    The threshold is peak_values / 2. We walk outward from the peak until
    the signal drops below the threshold (interpolating between samples
    for sub-bin precision), and return right_pos - left_pos.
    """
    n = len(signal)
    peak = signal[peak_idx]
    threshold = peak / 2.0

    left_hits = np.flatnonzero(signal[:peak_idx] < threshold)
    if len(left_hits) > 0:
        left = left_hits[-1]
        y0,y1 = signal[left], signal[left + 1]
        left_p = left + (threshold - y0) / (y1 - y0)
    else:
        left_p = 0.0

    right_hits = np.flatnonzero(signal[peak_idx+1:] < threshold)
    if len(right_hits) > 0:
        right = peak_idx + 1 + right_hits[0]
        y0,y1 = signal[right - 1], signal[right]
        right_p = right - 1 + (y0 - threshold) / (y0 - y1)
    else:
        right_p = n - 1

    return right_p - left_p

def _components(
        X: np.ndarray, 
        p: float = 1.0, 
        phi: float = 0.0, 
        axis: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the generalized rectangular components of circular data.
    Essentially the projections of a vector onto perpendicular axes.
    $V_x = V cos(\theta)$ and $V_y = V sin(\theta)$.
    """
    C = np.sum(np.cos(p * (X - phi)), axis=axis)
    S = np.sum(np.sin(p * (X - phi)), axis=axis)
    return C, S

def rayleightest(X: np.ndarray, axis: Optional[int] = None) -> np.ndarray|float:
    """Taken from https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html#rayleightest
    because I don't want to install a full package.
    """
    n = np.size(X, axis=axis)
    C,S = _components(X, 1.0, 0.0, axis=axis)
    Rbar = np.hypot(S,C)
    z = n * Rbar * Rbar
    tmp = 1.0
    if n < 50:
        tmp = 1 + (2 * z - z**2) / (4 * n) \
            - (24*z - 132 * z**2 + 76 * z**3 - 9 * z**4) \
            / (288 * n**2)

    pval = np.exp(-z) * tmp
    return pval
