import jax 
import jax.numpy as jnp
import numpy as np 

@jax.jit
def mT(x: np.ndarray|jax.Array):
    """Shorthand for np.matrix_transpose or jax.Array.mT"""
    if isinstance(x, jax.Array):
        return x.mT
    return np.matrix_transpose(x)

@jax.jit
def invmul(A: np.ndarray|jax.Array, B: np.ndarray|jax.Array):
    """Computes :math:`AB^{-1}`"""
    return mT(jnp.linalg.solve(mT(B), mT(A)))

@jax.jit
def mulinv(B: np.ndarray|jax.Array, A: np.ndarray|jax.Array):
    """Computes :math:`B^{-1}A`"""
    return jnp.linalg.solve(B, A)

@jax.jit
def logdet(x: np.ndarray|jax.Array):
    return jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(x))))