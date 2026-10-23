import torch
from .spiking_base import lag_spikes,count_spikes
from .diffusion import *

class DiffusionScaled(Diffusion):
    r"""Diffusion model scaled by spike count
        $$z_{t+1} = z_t + \sigma \sqrt{\Delta t \tau_E^{-1}}\xi_t$$
    where $\tau_e = c\cdot n_{t-1} + 1$
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        spikemat_train = kwargs['spikemat_train']
        spikemat_valid = kwargs.get('spikemat_valid', None)
        self.spikecount = {
            'train': None,
            'valid': None,
            'test' : None
        }
        self.spikecount['train'] = [count_spikes(spk) for spk in spikemat_train]
        if spikemat_valid is not None:
            self.spikecount['valid'] = [count_spikes(spk) for spk in spikemat_valid]

        self.tau = self.random_initializer(
            "tau", 
            (1,), 
            method=self.initialization_method
        )
        self.n_parameters += 1

    def name(self):
        return "Diffusion + Spike Scaling"

    def __str__(self):
        return (
            super().__str__() 
            + "\n"+rf"    $\tau$ = {self.tau.item():2f}"
        )

    def build_denom(self, tau: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        n_t = lag_spikes(spikes)
        return hseu.atleast_3d(1.0 + tau * n_t)

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        spikes = self.spikecount[mode]
        Q = super()._construct_transition_covariance(mode)
        Qs = []
        for spk in spikes:
            denom = self.build_denom(self.tau.exp(), spk)
            Qs.append(Q / denom)
        return Qs

    def build_batch_parameters(self, batch: int, mode: str = 'train') -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_covariance = self.global_parameters.transition_covariance[batch]
        return params

    def _solve_parameters(
            self,
            values: LDSResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
        ) -> torch.Tensor:
        params = [
            hseu.grad_tensor(self.diffusion),
            hseu.grad_tensor(self.tau)
        ]

        def loss_closure(params, stats: SufficientStatistics):
            diffusion = torch.exp(params[0])
            tau       = torch.exp(params[1])

            Ezz = stats.Ezz
            Ezz1 = stats.Ezz1
            sigma = diffusion**2 * self.dt

            total_loss = 0
            for i in range(len(Ezz)):
                T = len(Ezz[i])
                denom = self.build_denom(tau, self.spikecount['train'][i])[1:]
                _sig = sigma / denom
                loss = Ezz[i][1:] - Ezz1[i] - Ezz1[i].mT + Ezz[i][:-1]
                loss = torch.sum(loss / _sig, axis=0)
                total_loss += (
                    self.latent_dim * torch.sum(torch.log(_sig))
                    + torch.trace(loss)
                )
            return total_loss

        loss,params = hseu.optimize(
            loss_closure, 
            params,
            {
                'stats' : stats,
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )
        self.diffusion = params[0].detach()
        self.tau       = params[1].detach()
        self._initialize_globals()
        return loss

    def transform(self, spikemats: list[hseu.NDArray], X: list[hseu.NDArray]|None = None) -> LDSResults:
        self.spikecount['test'] = [count_spikes(spk) for spk in spikemats]
        return super().transform(spikemats, X)