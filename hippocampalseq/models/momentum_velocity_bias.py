import torch
import numpy as np
from typing import Callable, Any

import hippocampalseq.utils as hseu
from .momentum import Momentum

class MomentumVelocityBias(Momentum):
    r"""Momentum subclass that adds an additive bias to the hidden velocity.
    The bias is $F(\hat{v})$ where $\hat{v}$ is the true velocity provided to the
    function.
    """

    def __init__(
            self, 
            velocity: list[np.ndarray|torch.Tensor],
            bias_fn: tuple[Callable[[Any], Any],list[torch.Tensor]]|str = 'linear', 
            *args, 
            **kwargs
        ):
        r"""Create the MomentumVelocityBias model.
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
            elif bias_fn == 'exp':
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

        self.n_parameters += sum(p.numel() for p in self.bias_params)
        self.velocity = self._initialize_observations(velocity)

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
        params.transition_bias = self.global_parameters.transition_bias[batch]
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

        def loss_closure(
                params, 
                n_batches: int, 
                stats: SufficientStatistics,
                velocity: list[torch.Tensor]
            ):
            decay, diffusion = params[:2]
            bias_params = params[2:]

            Ez   = stats.Ez
            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1

            lmb = torch.exp(decay)
            sig = torch.exp(diffusion)
            M = 1 - lmb * self.dt
            sigma = sig**2 * self.dt
            v0 = sigma / (1 - M**2)

            total_loss = 0.0
            for b in range(n_batches):
                T = len(Ezz[b])
                bias = self.bias_fn(*bias_params, velocity[b])

                iloss = Ezz[b][0] / v0
                iloss = self.latent_dim * torch.log(v0) + iloss

                loss = Ezz[b][1:] - 2 * M * Ezz1[b] + M**2 * Ezz[b][:-1]
                loss = torch.sum(loss, axis=0) / sigma
                loss = self.latent_dim * (T-1) * torch.log(sigma) + loss

                ibloss = bias[0].mT @ bias[0] / v0

                bloss = 2 * M * bias[1:].mT @ Ez[b][1:]
                bloss = bloss - (2 * bias[1:].mT @ Ez[b][:-1]) 
                bloss = bloss + (bias[1:].mT @ bias[1:]) 
                bloss = torch.sum(bloss, axis=0) / sigma

                total_loss += (iloss + loss + ibloss + bloss) / 2

            return total_loss

        loss,params = hseu.optimize(
            loss_closure, 
            params,
            {
                'n_batches' : len(values.observations), 
                'stats'     : stats,
                'velocity'  : self.velocity
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