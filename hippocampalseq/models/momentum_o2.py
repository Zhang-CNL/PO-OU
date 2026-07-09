import torch
import numpy as np

import hippocampalseq.utils as hseu
from .momentum import *

class MomentumO2(Momentum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_dim    = 2
        self.augmented_dim = self.latent_dim * 2

        self.initial_diffusion = torch.rand(1)
        self.n_parameters      += 1

    def _construct_init_mean(self) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.augmented_dim, self.latent_dim)
        left = torch.cat((I, I), dim=0)
        return torch.cat((left, Z), dim=1)

    def _construct_init_var(self, initial_diffusion: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        init_cov = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_cov[:self.latent_dim, :self.latent_dim] = initial_diffusion**2 * self.dt * I
        return init_cov

    def _construct_transition_mat(self, decay: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        ex     = torch.exp(-decay * self.dt)
        A1     = I * (1 + ex)
        A2     = I * ex
        top    = torch.cat((A1, A2), dim=1)
        bottom = torch.cat((I, Z), dim=1)
        A = torch.cat((top, bottom), dim=0)
        return A

    def _construct_transition_cov(self, decay: torch.Tensor, diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        Q = (diffusion * self.dt) ** 2 / (2*decay) \
            * (1 - torch.exp(-2*decay * self.dt)) * I
        top    = torch.cat((Q, Z), dim=1)
        bottom = torch.cat((Z, I * jitter), dim=1)
        Gamma = torch.cat((top, bottom), dim=0)
        return Gamma

    def _init_priors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        Meanwhile, $P(z_2|z_1) = \mathcal{N}(z_2|z_1, \sigma_0^2 dt)$: a simple gaussian.

        Returns:
            (torch.Tensor): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): Prior covariance for augmented state
        """
        # $z_2 = I z_1 + \sigma_0^2dt\xi_1$
        init_mean = self._construct_init_mean()
        init_cov  = self._construct_init_var(self.initial_diffusion.exp())
        return init_mean, init_cov

    def _init_transition_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(self.decay.exp())
        Q = self._construct_transition_cov(self.decay.exp(), self.diffusion.exp()) 

        return A,Q

    def _init_observation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        H = torch.hstack((I, Z))
        R = self.approximate_covariance
        return H,R

    def _filter_init(self, values: MomentumResults, batch: int) -> MomentumResults:
        """
        mu1 = values.observations[batch][0]
        P1  = self.observation_covariance[batch][0]

        """
        mu0,P0 = super()._init_priors()
        # C is an identity matrix here, so we can ignore it.
        K1  = hseu.invmul(P0, P0 + self.observation_covariance)
        mu1 = mu0 + K1 @ (values.observations[batch][0] - mu0)
        P1  = (torch.eye(self.latent_dim) - K1) @ P0
        #"""

        # Filtered mean and covariance are augmented state spaces
        # $s_t = (z_t, z_{t-1})^T$
        """
        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = P1
        """
        values.filtered_mean[batch][0]  = mu1.repeat(self.latent_dim,1)
        values.filtered_cov[batch][0]   = P1.repeat(self.latent_dim,self.latent_dim)
        #"""
        values.predicted_mean[batch][0] = self.initial_state_mean @ values.filtered_mean[batch][0]
        values.predicted_cov[batch][0]  = self.initial_state_mean @ values.filtered_cov[batch][0] @ self.initial_state_mean.T + self.initial_state_covariance

        # Now we can calculate it for $P(z_1|z_0)$
        return values

    def filter(self, values: MomentumResults) -> MomentumResults:
        obs_cov     = self.observation_covariance
        transitions = self.transition_matrices
        trans_cov   = self.transition_covariance

        for batch in range(len(values.observations)):
            self.observation_covariance = obs_cov[batch][0]
            self.transition_matrices    = self.initial_state_mean
            self.transition_covariance  = self.initial_state_covariance
            values = self._filter_init(values, batch)

            for t in range(1, len(values.observations[batch])):
                self.transition_matrices    = transitions
                self.transition_covariance  = trans_cov
                self.observation_covariance = obs_cov[batch][t]
                values = self._filter(values, batch, t)

        self.observation_covariance = obs_cov
        self.transition_matrices    = transitions
        self.transition_covariance  = trans_cov
        return values

    def smooth(self, values: MomentumResults) -> MomentumResults:
        transitions = self.transition_matrices

        for batch in range(len(values.observations)):
            values = self._smooth_init(values, batch)
            self.transition_matrices = transitions

            for t in reversed(range(1, len(values.observations[batch]) - 1)):
                values = self._smooth(values, batch, t)

            self.transition_matrices = self.initial_state_mean
            values = self._smooth(values, batch, 0)

        self.transition_matrices = transitions
        return values

    def _loglikelihood(self, values: MomentumResults, _: KalmanStatistics) -> torch.Tensor:
        r"""Calculate the log-likelihood for this model given the current state.
        The full log-likelihood is:
        $$ln\ P(X,Z|\theta) = ln\ P(z_1) + ln\ P(z_2|z_1,\sigma_0) + \sum_{t=3}^T P(z_t|z_{t-1},\sigma,\lambda)
            + \sum_{t=1}^T ln\ P(x_t|z_t)
        $$
        Args:
            values (MomentumResults): Current state of the model. 
            _ (KalmanStatistics): Not used

        Returns:
            torch.Tensor: The log-likelihood for the model
        """
        return 0
        loglike = 0

        A = self.transition_matrices
        C = self.observation_matrices
        mu0 = self.initial_state_mean
        Gamma = self.transition_covariance
        Sigma = self.observation_covariance
        P0    = self.initial_state_covariance
        im,ic = super()._init_priors()

        rank = self.obs_dim

        log2pi = torch.log(2*PI)

        for b in range(len(values.observations)):
            T = len(values.observations[b])
            _loglike = 0

            muhat = values.smoothed_mean[b]
            vhat  = values.smoothed_cov[b]
            Jhat  = values.smoothed_gain[b]

            Exx = values.observations[b] @ values.observations[b].mT 
            Ezz = vhat + muhat @ muhat.mT 
            Ezz1 = Jhat[:-1] @ vhat[1:] + muhat[1:] @ muhat[:-1].mT

            illc = torch.logdet(ic) + rank * log2pi

        return 0

    def _solve_parameters(self,
            values: MomentumResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3,
        ) -> torch.Tensor:

        decay             = torch.zeros(1, requires_grad=True)
        diffusion         = torch.zeros(1, requires_grad=True)
        initial_diffusion = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            initial_diffusion.copy_(self.initial_diffusion)

        params = [decay, diffusion, initial_diffusion]

        def loss_closure(params, closure_kwargs):
            decay, diffusion, initial_diffusion = params
            diffusion         = torch.exp(diffusion)
            decay             = torch.exp(decay)
            initial_diffusion = torch.exp(initial_diffusion)

            n_batches = closure_kwargs['n_batches']
            stats     = closure_kwargs['stats']

            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1

            v1    = initial_diffusion**2 * self.dt
            alpha = 1 + torch.exp(-decay * self.dt)
            gamma = (diffusion * self.dt)**2 / (2 * decay) * (1 - torch.exp(-2 * decay * self.dt))
            total_loss = 0
            for i in range(n_batches):
                loss = 0. 
                T = len(Ezz[i])

                ill = Ezz[i][1] - 2 * Ezz1[i][0] + Ezz[i][0]
                ill = ill / v1
                ill = torch.log(v1**2) + ill
                loss += ill

                tll = Ezz[i][2:] - 2 * alpha * Ezz1[i][1:] + alpha**2 * Ezz[i][1:-1]
                tll = torch.sum(tll,axis=0) / gamma
                tll = (T-2) * torch.log(gamma**2) + tll
                loss += tll

                total_loss += loss / 2
            return total_loss

        loss,params = hseu.optimize(
            loss_closure,
            params,
            {
                'n_batches': len(values.observations), 
                'stats': stats
            },
            {
                'optimizer': optimizer,
                'lr': lr,
                'n_epochs': n_epochs,
                'gd_tol': gd_tol 
            }
        )
        self.decay             = params[0].detach()
        self.diffusion         = params[1].detach()
        self.initial_diffusion = params[2].detach()

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return loss
