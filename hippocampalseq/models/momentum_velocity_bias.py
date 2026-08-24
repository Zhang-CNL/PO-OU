import torch
import numpy as np
from typing import Callable, Any

import hippocampalseq.utils as hseu
from .momentum import Momentum

class MomentumVelocityBias(Momentum):
    f"""Momentum subclass that adds an additive bias to the hidden velocity.
    The bias is $F(\hat{v})$ where $\hat{v}$ is the true velocity provided to the
    function.
    """

    def __init__(
            self, 
            velocity: list[np.ndarray],
            bias_fn: tuple[Callable[[Any], Any],list[torch.Tensor]]|str = 'linear', 
            *args, 
            **kwargs
        ):
        f"""Create the MomentumVelocityBias model.
        Same as the momentum model, only the velocity dynamics have 
        a bias function applied:
        $$\dot{v}_t = -\lambda v_t + F(v_{true,t}) + \sigma \xi_t$$

        Args:
            velocity (list[np.ndarray]): True animal velocity for each session.
            bias_fn (tuple[Optional[Callable[[Any], Any]],list[torch.Tensor]]|str, optional): 
                The bias function to use. Must be differentiable if you want to solve for the parameters.
                Can be a string, or a tuple of a function and a list of parameters 
                to optimize. If it's a function, must be able to be called with f(*bias_params, velocity).
                If it's a string, it must be a valid default option. Defaults to 'linear'.
        """
        super().__init__(*args, **kwargs)

        if isinstance(bias_fn, str):
            if bias_fn == 'linear':
                self.bias_fn = lambda A,b,v: A @ v + b
                self.bias_params = [
                    torch.rand(self.latent_dim, self.latent_dim),
                    torch.rand(self.latent_dim, 1)
                ]
            elif bias_fn == 'glm':
                self.bias_fn = lambda A,b,v: torch.exp(A @ v + b)
                self.bias_params = [
                    torch.rand(self.latent_dim, self.latent_dim),
                    torch.rand(self.latent_dim, 1)
                ]
            else:
                raise ValueError(f"Unknown bias function: {bias_fn}")
        elif isinstance(bias_fn, (tuple,list)) and callable(bias_fn[0]):
            self.bias_fn = bias_fn[0]
            self.bias_params = bias_fn[1]
        else:
            raise TypeError("bias_fn must be a string or a tuple of a function and a list of parameters to optimize.")

        self.n_parameters += np.prod([ p.numel() for p in self.bias_params ])

    def _construct_transition_bias(self):
        bias = []
        for v in self.velocity:
            bias_top = hseu.atleast_2d(self.bias_fn(*self.bias_params, v))
            bias_bottom = torch.zeros_like(bias_top)
            bias.append(
                torch.hstack((bias_top, bias_bottom))
            )
        return bias

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        params = super().build_batch_parameters(batch)
        params.transition_bias = self.transition_bias[batch]
        return params

    def _solve_parameters(
            self, 
            values: MomentumResults, 
            stats: SufficientStatistics, 
            optimizer: str = "Adam", 
            lr: float = 0.01, 
            n_epochs: int = 1000, 
            gd_tol: float = 0.001
        ) -> torch.Tensor:
        
        decay = torch.zeros(1,requires_grad=True)
        diffusion = torch.zeros(1, requires_grad=True)
        bias_params = [torch.zeros_like(p, requires_grad=True) for p in self.bias_params]
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            for bp,p in zip(bias_params, self.bias_params):
                bp.copy_(p)

        params = [decay, diffusion] + bias_params

        def loss_closure(params, n_batches: int, stats: SufficientStatistics):
            decay, diffusion = params[:2]
            bias_params = params[2:]
            return 0.0

        loss,params = hseu.optimize(
            loss_closure, 
            params,
            {
                'n_batches' : len(values.observations), 
                'stats'     : stats,
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )

        self.decay = params[0].detach()
        self.diffusion = params[1].detach()
        self.bias_params = [p.detach() for p in params[2:]]

        self._initialize_globals()

        return loss