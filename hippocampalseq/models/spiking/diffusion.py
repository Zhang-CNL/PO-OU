import torch
import hippocampalseq.utils as hseu
from .spiking_base import *

class Diffusion(SpikingBase):
    r"""Basic diffusion model:
    $$z_{t+1} = z_t + \sigma \sqrt{\Delta t} \xi_t$$
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            1,
            slice(None,None),
            *args,
            **kwargs
        )
        self.diffusion = self.random_initializer("diffusion", (1,), method=self.initialization_method)
        self.n_parameters = 1

    def name(self):
        return "Diffusion"

    def __str__(self):
        return (
            self.name()
            + f"\n    Diffusion = {self.diffusion.exp().item():2f}"
        )

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        return torch.eye(self.augmented_dim)

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        return torch.eye(self.augmented_dim) * self.diffusion.exp()**2 * self.dt

    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        return torch.zeros((self.augmented_dim, 1))

    def _construct_emission_matrix(self, mode: str = "train") -> torch.Tensor:
        return torch.eye(self.emission_dim)

    def _construct_emission_bias(self, mode: str = "train") -> torch.Tensor:
        return torch.zeros((self.emission_dim, 1))

    def _construct_initial_mean(self, mode: str = "train") -> torch.Tensor:
        """Construct uniform prior approximated as a gaussian."""
        diffs = torch.tensor([es[1] + es[0] for es in self.environment_size])
        starts = torch.tensor([es[0] for es in self.environment_size])
        return (diffs / 2 + starts)[:,None]

    def _construct_initial_covariance(self, mode: str = "train") -> torch.Tensor:
        diffs = torch.tensor([es[1] + es[0] for es in self.environment_size])
        return torch.diag(diffs)**2 / 12
    
    def _solve_parameters(
            self,
            values: LDSResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
        ) -> torch.Tensor:
        params = [hseu.grad_tensor(self.diffusion)]

        def loss_closure(params, stats: SufficientStatistics):
            diffusion = torch.exp(params[0])

            Ezz = stats.Ezz
            Ezz1 = stats.Ezz1
            sigma = diffusion**2 * self.dt

            total_loss = 0
            for i in range(len(Ezz)):
                T = len(Ezz[i])
                loss = torch.sum(
                    Ezz[i][1:] - Ezz1[i]- Ezz1[i].mT + Ezz[i][:-1],
                    axis=0
                )
                total_loss += (
                    (T-1) * self.latent_dim * torch.log(sigma)
                    + torch.trace(loss) / sigma
                )
            return total_loss

        loss,params = hseu.optimize(
            loss_closure, 
            params,
            {
                'stats'     : stats,
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )
        self.diffusion = params[0].detach()
        self._initialize_globals()
        return loss