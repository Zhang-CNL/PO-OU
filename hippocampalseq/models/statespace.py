import numpy.typing as npt
from typing import Protocol, runtime_checkable

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

    def fit(self,
            X: npt.ArrayLike, 
            n_iter: int = 100, 
            emtol: float = 1e-3,
            maximization_type: str = 'em',
        ) -> StateSpaceResults:
        raise NotImplementedError