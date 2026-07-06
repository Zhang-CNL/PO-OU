import torch
import numpy as np
import numpy.typing as npt
from typing import (
    Protocol,
    runtime_checkable,
    Tuple,
    Any,
    Callable,
    Dict,
    List
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

    def _optimize(
        self, 
        closure: Callable[[List[torch.Tensor], Dict[str, Any]], torch.Tensor], 
        parameters: List[torch.Tensor], 
        closure_kwargs: Dict[str, Any],
        autograd_kwargs: Dict[str, Any]
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Optimize a given set of variables using pytorch's autograd feature.

        Args:
            closure (Callable[[List[torch.Tensor], Dict[str, Any]], torch.Tensor]): A function used to compute the log-likelihood.
            parameters (List[torch.Tensor]): The parameters to optimize. Passed as the first argument to `closure()`.
            closure_kwargs (Dict[str, Any]): Keyword arguments for the loss function. Passed as the second argument to `closure()`.
            autograd_kwargs (Dict[str, Any]): Keyword arguments for the optimizer.

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


    def fit(self,
            X: npt.ArrayLike, 
            *_: Tuple[Any,...],
        ) -> StateSpaceResults:
        raise NotImplementedError