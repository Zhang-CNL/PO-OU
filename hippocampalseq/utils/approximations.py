import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from typing import Tuple

from hippocampalseq.utils import atleast_2d

def analytical_gaussian_approximation(
        z: jax.Array, 
        pz: jax.Array, 
        bin_size: int
    ) -> Tuple[jax.Array, jax.Array]:
    B, Nx, Ny = pz.shape
    if z.ndim == 2:
        z = jnp.broadcast_to(z[None,...],(B,)+z.shape)

    D = z.shape[-1]
    
    mu = jnp.sum(pz.reshape(B, Nx*Ny, 1) * z, axis=1) / jnp.sum(pz,axis=(1,2))[:,None]
    
    z_centered = z - jnp.expand_dims(mu,1) # (B, N, D)
    z_centered = jnp.expand_dims(z_centered,-1)
    
    outer_products = z_centered @ jnp.transpose(z_centered, axes=(0,1,3,2))
    sigma = jnp.sum(pz.reshape((B, Nx*Ny, 1, 1)) * outer_products, axis=1) # (B, D, D)
    
    return jnp.expand_dims(mu,-1), sigma
