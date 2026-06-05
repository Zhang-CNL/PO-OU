import numpy as np
import torch 
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from hippocampalseq.utils import atleast_2d

def _to_cholesky(L_raw: torch.Tensor) -> torch.Tensor:
    """
    Map an unconstrained (B, D, D) matrix to a valid lower-triangular
    Cholesky factor with strictly positive diagonal via softplus.
    """
    L = torch.tril(L_raw)                                        # zero upper triangle
    diag_pos = F.softplus(L.diagonal(dim1=-2, dim2=-1))          # positive diagonal
    # Replace diagonal in-place-safe manner
    L = L - torch.diag_embed(L.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(diag_pos)
    return L

def analytical_gaussian_approximation(z, pz, bin_size: int):
    B, Nx, Ny = pz.shape
    if z.ndim == 2:
        z = z.unsqueeze(0).expand(B,-1,-1)        

    D = z.shape[-1]
    
    idx = torch.argmax(pz.reshape(B,-1), axis=1)
    row,col = torch.unravel_index(idx, pz.shape[1:])
    mu = torch.column_stack((col,row))
    mu = mu * bin_size + bin_size / 2
    #mu = torch.sum(pz.reshape(B, Nx*Ny, 1) * z, dim=1) # (B, D)
    
    z_centered = z - mu.unsqueeze(1) # (B, N, D)
    z_centered = z_centered.unsqueeze(-1)
    
    outer_products = z_centered @ z_centered.transpose(-2,-1)
    sigma = torch.sum(pz.view(B, Nx*Ny, 1, 1) * outer_products, dim=1) # (B, D, D)
    sigma /= Nx*Ny
    
    return mu.unsqueeze(-1), sigma

#def laplacian_approximation(z: torch.Tensor, pz: torch.Tensor, kl: str = "pq", lr: float = .01, n_epochs: int = 1000):
def iterative_gaussian_approximation(z, pz, bin_size: int, kl: str = "pq", lr: float = .01, n_epochs: int = 1000):
    r"""Laplacian approximation for the parameters of a Gaussian distribution.
    Finds the maximum point of the distribution $P(z)$ and then optimizes for the 
    value of $\Sigma$ that minimizes the KL divergence between $P(z)$ and $Q(z)$

    Args:
        z (torch.Tensor): The data points.
        pz (torch.Tensor): The probability distribution of the data points.
        kl (str): The type of KL divergence to use, either "pq" (optimize $KL(P||Q)$) or "qp" (optimize $KL(Q||P)$).
        lr (float): The learning rate for the optimization.

    Returns:
        mu (torch.Tensor): The mean of the Gaussian distribution.
        sigma (torch.Tensor): The covariance matrix of the Gaussian distribution.
    """
    assert kl in ["pq", "qp"]
    n_dims = z.shape[1] if z.ndim > 1 else 1

    mu = torch.unravel_index(torch.argmax(pz), pz.shape)
    mu = torch.tensor(mu, dtype=torch.double)[:,None]
    mu = torch.flip(mu, dims=(0,))
    mu = mu * bin_size + bin_size / 2

    pz = pz.ravel()
    lpz = torch.log(pz + 1e-12)
    csigma = torch.eye(n_dims, requires_grad=True)

    optimizer = torch.optim.Adam(
        [csigma], lr=lr
    )
    prev_entropy = np.inf

    for i in range(n_epochs):
        sigma = csigma @ csigma.T 

        mvn = MultivariateNormal(mu.ravel(), sigma)
        lqz = mvn.log_prob(z)
        if kl == "pq":
            entropy = -torch.sum(pz * lqz)
        else:
            qz = torch.exp(lqz)
            qz = qz / qz.sum()
            entropy = torch.sum(qz * (lqz - lpz))

        if i > 0 and abs(entropy.item() - prev_entropy) < .001:
            break
        prev_entropy = entropy.item()

        entropy.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    
    mu = torch.reshape(mu, (n_dims, 1)).detach()
    sigma = (csigma @ csigma.T).detach()
    return mu,sigma,entropy

def calc_normal_params(z: np.ndarray, pz: np.ndarray, dz: float|tuple):
    """Calculate normal parameters from a multivariate normal distribution.

    Args:
        z (np.ndarray): (n_points, n_dims) grid points for the distribution.
        pz (np.ndarray): (n_points,) probability values for the distribution.
        dz (float|tuple): Grid spacing in each dimension.

    Returns:
        (np.ndarray): (n_dims,) mean vector.
        (np.ndarray): (n_dims, n_dims) covariance matrix.
        (float): Normalization constant.
    """
    nd = z.shape[1]
    if type(dz) is float:
        dz = dz**nd
    else:
        dz = np.prod(dz)
    w = atleast_2d(pz * dz)

    Ez = np.einsum('ij,ij->j', z, w)
    Ez2 = np.einsum('ij,ik,i->jk', z, z, w.ravel())
    Vz = Ez2 - np.outer(Ez, Ez)
    Zt = np.sum(w) # Sanity check. Should sum to 1

    return Ez,Vz,Zt