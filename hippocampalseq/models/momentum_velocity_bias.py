import torch
from typing import Callable, Any

from .momentum import Momentum

class MomentumVelocityBias(Momentum):
    f"""Momentum subclass that adds an additive bias to the hidden velocity.
    The bias is $F(\hat{v})$ where $\hat{v}$ is the true velocity provided to the
    function.
    """

    def __init__(self, 
            bias_fn: tuple[Callable[[Any], Any],list[torch.Tensor]]|str = 'linear', 
            *args, 
            **kwargs
        ):
        f"""Create the MomentumVelocityBias model.
        Same as the momentum model, only the velocity dynamics have 
        a bias function applied:
        $$\dot{v}_t = -\lambda v_t + F(v_{true,t}) + \sigma \xi_t$$

        Args:
            bias_fn (tuple[Optional[Callable[[Any], Any]],list[torch.Tensor]]|str, optional): 
                The bias function to use. 
                Can be a string, or a tuple of a function and a list of parameters 
                to optimize. If it's a string, it must be a valid default option. Defaults to 'linear'.
        """
        super().__init__(*args, **kwargs)

        if isinstance(bias_fn, str):
            if bias_fn == 'linear':
                self.bias_fn = lambda A,v,b: A @ v + b
                self.bias_params = [
                    torch.rand(self.latent_dim, self.latent_dim),
                    torch.rand(self.latent_dim, 1)
                ]
            elif bias_fn == 'glm':
                self.bias_fn = lambda A,v,b: torch.exp(A @ v + b)
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
