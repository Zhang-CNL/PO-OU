import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

import hippocampalseq.utils as hseu
import hippocampalseq.models as hsem

__all__ = [
    'Momentum'
]


class Momentum(hsem.StateSpaceModel):
    def __init__(
            self,
            place_fields: np.ndarray|torch.Tensor, 
            spikemat: np.ndarray|torch.Tensor,
            dt: float, 
            bins: tuple,
            seed: int|None = 42
        ):
        """Initialize the momentum SSM.
        
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Ngrid) Flattened place fields.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            bins (tuple): Number of bins for each latent dimension.
            seed: (int|None): Seed for the random number generator
        """
        super().__init__(2, 2, 2)

        self.dt   = torch.tensor(dt)
        self.bins = bins 
        assert len(bins) == self.latent_dim, "Number of bins and latent dimensions must match"

        x = torch.arange(0, self.bins[0], 1)
        y = torch.arange(0, self.bins[1], 1)
        self.grid = hseu.bin_points(x,y)

        if seed is not None:
            torch.random.manual_seed(seed)

        self.emission_probabilities = hseu.calc_poisson_emission_probabilities_2d(
            torch.from_numpy(spikemat).double(), 
            torch.from_numpy(place_fields),
            self.dt
        )

        T = self.emission_probabilities.shape[0]
        self.approx_mean = torch.zeros(T, self.latent_dim, 1)
        self.approx_cov = torch.zeros(T, self.latent_dim, self.latent_dim)
        for t in range(T):
            self.approx_mean[t], self.approx_cov[t] = hseu.laplacian_approximation(
                self.grid,
                self.emission_probabilities[t]
            )

        # Random initialization of parameters
        self.initial_diffusion = torch.rand(1) * 1000
        self.decay             = torch.rand(1) * (800 - 1) + 1 
        self.diffusion         = torch.rand(1) * (400 - 40) + 40

        pz = torch.tensor([self.emission_probabilities[t][*self.approx_mean[t].to(int)] for t in range(T)])
        # A = torch.column_stack((self.approx_mean, torch.ones(T, 1, 1)))
        A = self.approx_mean
        A = A.squeeze(-1)
        print(A.shape, pz.shape)
        soln,resid,rank,s = torch.linalg.lstsq(A, pz)
        a,b = soln
        print(a,b)

        # TODO:
        # Instead of randomly initializing parameters,
        # fit a plane to parameters based on P(z_t|z_{t-1},z_{t-2})
        # Use approx_mean as z

        #print(self.initial_diffusion, self.decay, self.diffusion)

    def _construct_init_var(self, initial_diffusion: torch.Tensor, jitter=0.0):
        I = torch.eye(self.latent_dim)
        init_cov = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_cov[:self.latent_dim, :self.latent_dim] = initial_diffusion**2 * self.dt * I
        init_cov[self.latent_dim:, self.latent_dim:] = jitter * I
        return init_cov

    def _construct_transition_mat(self, decay: torch.Tensor, diffusion: torch.Tensor):
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        A1     = I * (1 + torch.exp(-decay * self.dt))
        A2     = I * torch.exp(-decay * self.dt)
        top    = torch.cat((A1, A2), dim=1)
        bottom = torch.cat((I, Z), dim=1)
        A = torch.cat((top, bottom), dim=0)
        return A

    def _construct_transition_cov(self, decay: torch.Tensor, diffusion: torch.Tensor, jitter=0.0):
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        Q = (diffusion * self.dt) ** 2 / (2*decay) * (1 - torch.exp(-2*decay * self.dt)) * I
        top    = torch.cat((Q, Z), dim=1)
        bottom = torch.cat((Z, I * jitter), dim=1)
        Gamma = torch.cat((top, bottom), dim=0)
        return Gamma

    def _init_priors(self) -> tuple:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        Meanwhile, $P(z_2|z_1) = \mathcal{N}(z_2|z_1, \sigma_0^2 dt)$: a simple gaussian.

        Returns:
            (torch.Tensor): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): Prior covariance for augmented state
        """
        I = torch.eye(self.latent_dim)
        # $z_2 = I z_1 + \sigma_0^2dt\xi_1$
        init_mean = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_mean[:self.latent_dim, :self.latent_dim] = I
        init_mean[self.latent_dim:, :self.latent_dim] = I
        init_cov = self._construct_init_var(self.initial_diffusion)
        return init_mean, init_cov

    def _init_transition_matrices(self) -> tuple:
        """Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(self.decay, self.diffusion)
        Q = self._construct_transition_cov(self.decay, self.diffusion) 

        return A,Q

    def _init_observation_matrices(self):
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        C = torch.hstack((I, Z))
        #H = np.zeros((self.augmented_dim, self.augmented_dim))
        H = self.approx_cov
        return C,H

    def _filter_init(self, values: hsem.KalmanResults):
        """
        Initialize the filter for the first observation.
        Since we have a uniform prior for this model, we use the information filter
        to handle it.
        """
        I = torch.eye(self.augmented_dim)
        # Deal with the uniform prior using the information filter
        # Since the uniform prior contains 0 precision, only the likelihood function contributes 
        sigma0i = torch.inverse(self.observation_covariance[0])
        omega0 = self.observation_matrices.T @ sigma0i @ self.observation_matrices
        xi0 = self.observation_matrices.T @ sigma0i @ values.observations[0]

        P0  = torch.inverse(omega0 + .00001 * torch.eye(self.augmented_dim))
        mu0 = P0 @ xi0
        P0  = 0.5 * (P0 + P0.T)

        # Filtered mean and covariance are augmented state spaces
        # $s_t = (z_t, z_{t-1})^T$
        values.filtered_mean[0]  = mu0
        values.filtered_cov[0]   = P0
        values.predicted_mean[0] = self.initial_state_mean @ mu0
        values.predicted_cov[0]  = self.initial_state_mean @ P0 @ self.initial_state_mean.T + self.initial_state_covariance

        # Now we can calculate it for $P(z_1|z_0)$
        return values

    def _filter(self, values: hsem.KalmanResults, t: int):
        """
        Run the Kalman filter for a single time step.
        Use our initial transition and covariance matrices for t == 0
        """
        
        A = self.transition_matrices if t > 1 else self.initial_state_mean
        C = self.observation_matrices
        gamma = self.transition_covariance if t > 1 else self.initial_state_covariance
        sigma = self.observation_covariance

        Am1 = values.predicted_mean[t-1]
        Pn1 = values.predicted_cov[t-1]

        PnCt = Pn1 @ C.T
        K = hseu.invmul(PnCt, C @ PnCt + sigma[t])

        mu_t = Am1 + K @ (values.observations[t] - C @ Am1)
        v_t = (torch.eye(self.augmented_dim) - K @ C) @ Pn1

        Am = A @ mu_t
        Pt = A @ v_t @ A.T + gamma
        Pt = .5 * (Pt + Pt.T)

        values.predicted_mean[t] = Am
        values.predicted_cov[t]  = Pt
        values.filtered_mean[t]  = mu_t
        values.filtered_cov[t]   = v_t

        return values

    def _smooth(self, values: hsem.KalmanResults, t: int):
        """
        Smooth the Kalman filter results for one timestep.

        If t == 0, use the initial state mean and covariance to calculate the smoothed values.
        Otherwise, use the previous smoothed values and the transition matrices to calculate the current smoothed values.
        """
        if t == 0:
            Amt = values.predicted_mean[t]
            Pt  = values.predicted_cov[t]

            J = hseu.invmul(values.filtered_cov[t] @ self.initial_state_mean.T , Pt + .00001 * torch.eye(self.augmented_dim))
            muht = values.filtered_mean[t] + J @ (values.smoothed_mean[t+1] - Amt) 
            vht = values.filtered_cov[t] + J @ (values.smoothed_cov[t+1] - Pt) @ J.mT

            values.smoothed_gain[t] = J
            values.smoothed_mean[t] = muht
            values.smoothed_cov[t]  = vht
            return values
        else:
            return super()._smooth(values, t)


    def _em_mle(self, values, stats, normalize):
        raise NotImplementedError("Maximum likelihood estimators not implemented for momentum models. Use autograd.")

    def _em_autograd(self, 
            values: hsem.KalmanResults,
            stats: hsem.SufficientStatistics,
            normalize: bool,
            lr: float = 1e-3, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
            seed: int|None = 42
        ) -> torch.Tensor:
        """Perform maximum likelihood estimation of all relevant parameters for the momentum SSM using autograd.

        Args:
            values (hsem.KalmanResults): Kalman filter results.
            stats (hsem.SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            normalize (bool): If True, normalize the transition and observation matrices.
            lr (float): Learning rate for the optimizer.
            n_epochs (int): Number of epochs for SGD.
            gd_tol (float): Tolerance for SGD.
            seed (int|None): Seed for the random number generator.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        if seed is not None:
            torch.random.manual_seed(seed)

        T = values.observations.shape[0]

        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        decay             = torch.zeros(1, requires_grad=True)
        diffusion         = torch.zeros(1, requires_grad=True)
        initial_diffusion = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            initial_diffusion.copy_(self.initial_diffusion)

        A = self._construct_transition_mat(decay, diffusion)
        Gamma = self._construct_transition_cov(decay, diffusion, jitter=0.00001)

        C = self.observation_matrices
        Sigma = self.observation_covariance

        init_mat = self.initial_state_mean
        init_cov = self._construct_init_var(initial_diffusion, jitter=0.00001)

        optimizer = torch.optim.Adam(
            [diffusion, decay, initial_diffusion],
            lr=lr
        )

        prev_loss = 0.
        bins = torch.tensor(self.bins)

        for epoch in range(n_epochs):
            loss = 0
            
            ill = stats.Ezz[1] - stats.Ezz1[0] @ init_mat.mT - init_mat @ stats.Ez1z[0] + init_mat @ stats.Ezz[0] @ init_mat.mT
            ill = torch.linalg.solve(init_cov, ill)
            loss += torch.trace(ill)

            tll = stats.Ezz[2:] - stats.Ezz1[1:] @ A.mT - A @ stats.Ez1z[1:]  + A @ stats.Ezz[1:-1] @ A.mT
            tll = torch.sum(tll, axis=0)
            tll = torch.linalg.solve(Gamma, tll)
            loss += torch.trace(tll)

            ell = stats.Exx - stats.Exz @ C.mT - C @ stats.Ezx + C @ stats.Ezz @ C.mT
            ell = torch.linalg.solve(Sigma, ell)
            ell = torch.sum(ell, axis=0)
            loss += torch.trace(ell)

            loss += torch.sum(torch.logdet(Sigma))
            loss += torch.logdet(init_cov)
            loss += torch.logdet(Gamma) * (T - 2)
            loss /= 2.0

            if epoch > 0 and abs((loss.item() - prev_loss) / prev_loss) < gd_tol: 
                break
            prev_loss = loss.item()
        
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        self.decay             = decay.detach()
        self.diffusion         = diffusion.detach()
        self.initial_diffusion = initial_diffusion.detach()
        #print(self.initial_diffusion, self.decay, self.diffusion)

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return torch.tensor(prev_loss)

    def em(self, X=None, **em_args):
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            X (torch.Tensor, optional): Value ignored. We fit to self.approx_mean.
            **em_args: Keyword arguments to pass to the parent class's em method.

        Returns:
            torch.Tensor: The negative log likelihood of the data given the model parameters.
        """
        X = self.approx_mean
        return super().em(X, **em_args)

