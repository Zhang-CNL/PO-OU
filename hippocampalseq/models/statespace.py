import torch
import numpy as np
import numpy.typing as npt
from typing import (
    Protocol,
    runtime_checkable,
)

__all__ = [
    'StateSpace',
    'StateSpaceResults',
    'SufficientStatistics'
]

@runtime_checkable
class SufficientStatistics(Protocol):
    pass 

@runtime_checkable
class StateSpaceResults(Protocol):
    pass

class StateSpace:
    def filter(self, values: StateSpaceResults) -> StateSpaceResults:
        raise NotImplementedError

    def smooth(self, values: StateSpaceResults) -> StateSpaceResults:
        raise NotImplementedError

    def bic(self, max_loglikelihood: float, n_observations: float) -> float:
        assert hasattr(self, 'n_parameters'), "Forgot to construct a parameter `n_parameters` in the class."
        return -2*max_loglikelihood + self.n_parameters * np.log(n_observations)
    
    def aic(self, max_loglikelihood: float) -> float:
        assert hasattr(self, 'n_parameters'), "Forgot to construct a parameter `n_parameters` in the class."
        return 2*(self.n_parameters - max_loglikelihood)

    def fit(self,
            X: np.ndarray,
            *_: tuple,
        ) -> StateSpaceResults:
        raise NotImplementedError