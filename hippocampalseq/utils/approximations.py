import numpy as np
import torch 
from torch.distributions import MultivariateNormal

from hippocampalseq.utils import atleast_2d

def laplacian_approximation(z: torch.Tensor, pz: torch.Tensor, kl:str = "pq", lr:float = .01, n_epochs:int = 1000):
    r"""Laplacian approximation for the parameters of a Gaussian distribution.
    Finds the maximum point of the distribution $P(z)$ and then optimizes for the 
    value of $\Sigma$ that minimizes the KL divergence between $P(z)$ and $Q(z)$

    Parameters:
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

    pz = pz.ravel()
    lpz = torch.log(pz)
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
            entropy = torch.sum(qz * (lqz - lpz))
            

        if i > 0 and abs(entropy.item() - prev_entropy) < .01:
            break
        prev_entropy = entropy.item()

        entropy.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    
    mu = torch.reshape(mu, (n_dims, 1)).detach()
    sigma = (csigma @ csigma.T).detach()
    return mu,sigma

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