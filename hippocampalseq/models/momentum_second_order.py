import torch
import numpy as np

import hippocampalseq.utils as hseu
from .momentum import *

class MomentumSecondOrder(Momentum):
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

    def _construct_init_var(self, initial_diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        init_cov = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_cov[:self.latent_dim, :self.latent_dim] = initial_diffusion**2 * self.dt * I
        init_cov[self.latent_dim:, self.latent_dim:] = jitter * I
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

    def _em_autograd(self, 
            values: MomentumResults,
            stats: SufficientStatistics,
            _: bool,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
            seed: int|None = 42
        ) -> torch.Tensor:
        """Perform maximum likelihood estimation of all relevant parameters for the momentum SSM using autograd.

        Args:
            values (MomentumResults): Momentum filtering pass results.
            stats (SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            _ (bool): Option to normalize the transition and observation matrices. Ignored.
            optimizer (str): The optimizer to use.
            lr (float): Learning rate for the optimizer.
            n_epochs (int): Number of epochs for SGD.
            gd_tol (float): Tolerance for SGD.
            seed (int|None): Seed for the random number generator.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        if seed is not None:
            torch.random.manual_seed(seed)

        optimizers = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'AdamW': torch.optim.AdamW,
            'LBFGS': torch.optim.LBFGS
        }

        I = torch.eye(self.latent_dim)

        decay             = torch.zeros(1, requires_grad=True)
        diffusion         = torch.zeros(1, requires_grad=True)
        initial_diffusion = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            initial_diffusion.copy_(self.initial_diffusion)

        optimizer = optimizers[optimizer](
            [diffusion, decay, initial_diffusion],
            lr=lr
        )

        prev_loss = np.inf
        btrace = torch.vmap(torch.trace)

        Ezz  = stats.Ezz
        Ezz1 = stats.Ezz1

        for epoch in range(n_epochs):
            def loss_closure():
                optimizer.zero_grad()
                _idiff = torch.exp(initial_diffusion)
                _diff  = torch.exp(diffusion)
                _decay = torch.exp(decay)

                v1 = _idiff**2 * self.dt
                alpha = 1 + torch.exp(-_decay * self.dt)
                gamma = (_diff * self.dt)**2 / (2 * _decay) * (1 - torch.exp(-2 * _decay * self.dt))
                total_loss = 0
                for i in range(len(values.observations)):
                    loss = 0. 
                    T = values.observations[i].shape[0]

                    ill = Ezz[i][1] - 2 * Ezz1[i][0] + Ezz[i][0]
                    ill = ill / v1
                    ill = torch.log(v1**2) + ill
                    loss += ill

                    tll = Ezz[i][2:] - 2 * alpha * Ezz1[i][1:] + alpha**2 * Ezz[i][1:-1]
                    tll = torch.sum(tll,axis=0) / gamma
                    tll = (T-2) * torch.log(gamma**2) + tll
                    loss += tll

                    total_loss += loss / 2
                total_loss.backward(retain_graph=True)
                return total_loss

            total_loss = optimizer.step(loss_closure)

            if epoch > 0 and abs((total_loss.item() - prev_loss) / prev_loss) < gd_tol: 
                break

            prev_loss = total_loss.item()
        
        self.decay             = decay.detach()
        self.diffusion         = diffusion.detach()
        self.initial_diffusion = initial_diffusion.detach()

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return torch.tensor(prev_loss)
