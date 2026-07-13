import torch
import numpy as np
import numpy.typing as npt
from typing import Optional
from dataclasses import dataclass

from .statespace import * 
import hippocampalseq.utils as hseu

@dataclass 
class SCAResults:
    u: npt.ArrayLike
    v: npt.ArrayLike
    z: npt.ArrayLike

class SCA(StateSpace):
    def __init__(
            self, 
            factor_dim: int, 
            sparse_weight: float, 
            orthogonal_weight: float,
        ):
        super().__init__()

        self.factor_dim = factor_dim
        self.sparse_weight = sparse_weight
        self.orthogonal_weight = orthogonal_weight


    def fit(
        self, 
        X: npt.ArrayLike, 
        init_type: str = "random",
        weights: Optional[npt.ArrayLike] = None, 
        **maximization_args
    ):
        assert len(X) == weights.shape[0] and len(X) == weights.shape[1],"Weights must have the shape (T,T)"
        assert len(X.shape) == 2, "X must be 2D (T,N)"
        X = torch.from_numpy(X)
        T,N = X.shape

        if weights is None:
            weights = np.ones((T,1))

        if init_type == "random":
            _U = hseu.orthog(torch.random.randn(T, self.factor_dim))
            _V = _U.T
        elif init_type == 'pca':
            _,_U,_ = hseu.pca(X, self.factor_dim)
            _V = _U.T 
        elif init_type == 'svd':
            _U,_,_V = torch.linalg.svd(X, full_matrices=False)
            _U = _U[:,:self.factor_dim]
            _V = _V[:,:self.factor_dim]
        else:
            raise Exception(f"Unknown initialization type {init_type}. Must be random, pca or svd.")
        
        U = torch.zeros(_U.shape, requires_grad=True)
        V = torch.zeros(_V.shape, requires_grad=True)
        with torch.no_grad():
            U.copy_(_U)
            V.copy_(_V)

        # SCA loss optimization
        def _sca(params, kwargs):
            U,V = params
            X  = kwargs["x"]
            W  = kwargs["w"]
            sw = kwargs["sw"]
            ow = kwargs["ow"]
            K  = kwargs["k"]
            T,N = x.shape
            I = torch.eye(K)

            Z = X @ U
            response_variance = torch.linalg.norm(W @ (X - Z @ V), ord='fro')**2

            sparse_time = torch.linalg.norm(Z, ord=1)

            orthogonal_space = torch.linalg.norm(V @ V.T - I, ord='fro')**2
            return response_variance \
                + sw * sparse_time \
                + ow * orthogonal_space

        params = [U,V]
        _,params = hseu.optimize(
            sca, 
            params,
            {
                "x" : X,
                "w" : weights,
                "sw": self.sparse_weight,
                "ow": self.orthogonal_weight,
                "k" : self.factor_dim,
            }, 
            maximization_args
        )
