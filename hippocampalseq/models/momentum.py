import os
import numpy as np
import numpy.typing as npt
import pynapple as nap
import torch
import warnings
from typing import List, Tuple
from dataclasses import dataclass, field
from scipy.optimize import least_squares

import hippocampalseq.utils as hseu

from .kalman_filter import *
__all__ = [
    'Momentum',
    'MomentumResults'
]

@dataclass
class MomentumResults(KalmanResults):
    emission_probabilities : List[torch.Tensor] = field(default_factory=list)
    approximate_mean       : List[torch.Tensor] = field(default_factory=list)
    approximate_covariance : List[torch.Tensor] = field(default_factory=list)

class Momentum(KalmanFilter):
    """State-space model that includes momentum as a parameters.
    Essentially, this collapses down to a second-order markov chain, so we can use kalman filtering.
    We have a uniform prior and our observation covariance shifts over time, so we take that into account here as well.
    """
    def __init__(
            self,
            place_fields: npt.ArrayLike, 
            spikemat: List[npt.ArrayLike],
            dt: float, 
            environment_size: tuple,
            bin_size: int, 
            seed: int|None = 42
        ):
        """Initialize the momentum SSM.
        
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (tuple): Size of the environment. (xmin, ymin, xmax, ymax)
            bin_size (int): Size of individual bins in cm.
            seed: (int|None): Seed for the random number generator
        """
        super().__init__(4, 2, 1)

        self.dt               = torch.tensor(dt)
        self.environment_size = environment_size
        self.bin_size         = bin_size
        #assert len(environment_size) == 2*self.latent_dim, "Environment shape and latent dimensions must match"

        self.grid = hseu.create_grid(self.environment_size, self.bin_size)

        if seed is not None:
            torch.random.manual_seed(seed)

        place_fields = torch.from_numpy(place_fields)


        values = MomentumResults(
            emission_probabilities = [],
            approximate_mean       = [],
            approximate_covariance = []
        )
        self.emission_probabilities = []
        self.approximate_mean       = []
        self.approximate_covariance = []
        for k,v in enumerate(spikemat):
            ep = torch.from_numpy(v).double()
            emission_probability = hseu.calc_poisson_emission_probabilities_2d(
                ep, 
                place_fields,
                self.dt
            )
            emission_probability /= torch.sum(emission_probability, axis=(1,2), keepdim=True)

            approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
                self.grid,
                emission_probability,
                self.bin_size
            )
            
            self.emission_probabilities.append(emission_probability)
            self.approximate_mean.append(approx_mean)
            self.approximate_covariance.append(approx_cov)


        # Random initialization of parameters
        # Scale of ln(10) meters
        self.decay        = torch.rand(1) 
        self.diffusion    = torch.rand(1) 
        self.n_parameters = 2

    def _initialize(self, X: torch.Tensor) -> MomentumResults:
        tf_base = [torch.ones((x.shape[0], self.augmented_dim, 1)) for x in X]
        cov_base = [torch.ones((x.shape[0], self.augmented_dim, self.augmented_dim)) for x in X]
        def _copy(x):
            return [i.clone() for i in x]
        res = MomentumResults(
            observations   = X,
            predicted_mean = _copy(tf_base),
            predicted_cov  = _copy(cov_base),
            filtered_mean  = _copy(tf_base),
            filtered_cov   = _copy(cov_base),
            smoothed_gain  = _copy(cov_base),
            smoothed_mean  = _copy(tf_base),
            smoothed_cov   = _copy(cov_base),
            emission_probabilities = self.emission_probabilities,
            approximate_mean       = self.approximate_mean,
            approximate_covariance = self.approximate_covariance
        )
        return res

    def _construct_transition_mat(self, decay: torch.Tensor) -> torch.Tensor:
        I  = torch.eye(self.obs_dim)
        Z  = torch.zeros(self.obs_dim, self.obs_dim)
        If = torch.eye(self.augmented_dim)

        M1 = -decay * I
        top = torch.cat((M1, Z), dim=1)
        bottom = torch.cat((I , Z), dim=1)
        A = torch.cat((top, bottom), dim=0) * self.dt + If
        return A

    def _construct_transition_cov(self, diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.obs_dim)
        Z = torch.zeros((self.obs_dim, self.obs_dim))

        sigma_m = diffusion * torch.sqrt(self.dt) * I
        top = torch.cat((sigma_m, Z), dim=1)
        bottom = torch.cat((I, I*jitter), dim=1)
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
        dx = self.environment_size[2] - self.environment_size[0]
        dy = self.environment_size[3] - self.environment_size[1]
        init_mean = torch.tensor([
            [self.environment_size[0] + dx/2],
            [self.environment_size[1] + dy/2]
        ])
        init_cov = torch.tensor([
            [dx**2 / 12, 0],
            [0, dy**2 / 12]
        ])
        return init_mean, init_cov

    def _init_transition_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(self.decay.exp())
        Q = self._construct_transition_cov(self.diffusion.exp()) 

        return A,Q

    def _init_observation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.obs_dim)
        Z = torch.zeros(self.obs_dim, self.obs_dim)
        H = torch.hstack((I, Z))
        R = self.approximate_covariance
        return H,R

    def _filter_init(self, values: MomentumResults, batch: int) -> MomentumResults:
        """Initialize the filter for the first observation.
        Since we have a uniform prior for this model, we use the information filter
        to handle it.

        Args:
            values (MomentumResults): Results from previous filter passes.
            batch (int): Batch index
        
        Results:
            (MomentumResults): Filtered results
        """
        """
        mu1 = values.observations[batch][0]
        P1  = self.observation_covariance[batch][0]
        """
        mu0 = self.initial_state_mean
        P0  = self.initial_state_covariance
        # C is an identity matrix here, so we can ignore it.
        K1  = hseu.invmul(P0, P0 + self.observation_covariance)
        mu1 = mu0 + K1 @ (values.observations[batch][0] - mu0)
        P1  = (torch.eye(self.obs_dim) - K1) @ P0
        #"""

        # Filtered mean and covariance are augmented state spaces
        # $s_t = (z_t, z_{t-1})^T$
        """
        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = P1
        """
        values.filtered_mean[batch][0]  = mu1.repeat(self.obs_dim,1)
        values.filtered_cov[batch][0]   = P1.repeat(self.obs_dim,self.obs_dim)
        #"""
        values.predicted_mean[batch][0] = self.transition_matrices @ values.filtered_mean[batch][0]
        values.predicted_cov[batch][0]  = self.transition_covariance @ values.filtered_cov[batch][0] @ self.transition_covariance.T + self.transition_matrices

        # Now we can calculate it for $P(z_1|z_0)$
        return values

    def filter(self, values: MomentumResults) -> MomentumResults:
        obs_cov = self.observation_covariance
        for batch in range(len(values.observations)):
            self.observation_covariance = obs_cov[batch][0]
            values = self._filter_init(values, batch)
            for t in range(1, len(values.observations[batch])):
                self.observation_covariance = obs_cov[batch][t]
                values = self._filter(values, batch, t)
        self.observation_covariance = obs_cov
        return values

    def _calc_sufficient_stats(self, values: MomentumResults) -> KalmanStatistics:
        """
        Calculate sufficient statistics for performing maximization given the filtered
        and smoothed values of the model.
        Avoids calculating various unused values from the Kalman filtering version.

        Args:
            values (MomentumResults): The filtered and smoothed values of the model.

        Returns:
            KalmanStatistics: The sufficient statistics of the model.
        """
        btrace = torch.vmap(torch.trace)
        Cov  = [
            (sc[1:,:self.latent_dim,:self.latent_dim] @ sg[:-1,:self.latent_dim,:self.latent_dim].mT)
                for sc,sg in zip(values.smoothed_cov, values.smoothed_gain)
        ]
        Ez   = [ez[:,:self.latent_dim] for ez in values.smoothed_mean]
        Ezz  = [
            (sm[:,:self.latent_dim].mT @ sm[:,:self.latent_dim]).squeeze() 
                for sm in values.smoothed_mean
        ]
        Ezz1 = [
            btrace(c).squeeze() + (sm[1:,:self.latent_dim].mT @ sm[:-1,:self.latent_dim]).squeeze()
                for c,sm in zip(Cov,values.smoothed_mean)
        ]
        return KalmanStatistics(
            Cov=Cov,
            Ez=Ez,
            Ezz=Ezz,
            Ezz1=Ezz1,
            Ez1z=None,
            Exx=None,
            Exz=None,
            Ezx=None
        )

    def _loglikelihood(self, values: MomentumResults, stats: KalmanStatistics) -> torch.Tensor:
        """Calculate the log-likelihood for the model given the current state.
        
        Args:
            values (MomentumResults): The current decoded values for hidden states.
            stats (KalmanStatistics): Sufficient statistics calculated from these values.

        Returns:
            torch.Tensor: The log-likelihood for the model.
        """
        #ell = stats.Exx[b] - stats.Exz[b] @ C.mT - C @ stats.Ezx[b] + C @ stats.Ezz[b] @ C.mT
        #ell = torch.linalg.solve(Sigma[b] + .00001 * torch.eye(self.obs_dim), ell)
        #ell = torch.sum(ell, axis=0)
        #_ll += torch.trace(ell) 

        ll = 0
        idiff = torch.exp(self.initial_diffusion)
        diff  = torch.exp(self.diffusion)
        decay = torch.exp(self.decay)

        Sigma = self.observation_covariance
        v1 = idiff**2 * self.dt 
        alpha = 1 + torch.exp(-decay * self.dt)
        gamma = (diff * self.dt)**2 / (2*decay) * (1 - torch.exp(-2*decay * self.dt))

        for b in range(len(values.observations)): 
            T = values.observations[b].shape[0]
            _ll = 0
            _ll += torch.log(v1**2)
            _ll += torch.log(gamma**2) * (T-2)
            #_ll += torch.sum(torch.logdet(Sigma[b]))

            ill = stats.Ezz[b][1] - 2*stats.Ezz1[b][0] + stats.Ezz[b][0]
            ill = ill / v1 
            _ll += ill

            tll = stats.Ezz[b][2:] - 2*alpha*Ezz1[b][1:] + alpha**2 * stats.Ezz[b][1:-1]
            tll = torch.sum(tll, axis=0) / gamma 
            _ll += tll 

            #ell = stats.Exx[b] - stats.Ezx[b] - stats.Ezx[b].mT + stats.Ezz[b]

            ll += _ll / 2 + T * self.augmented_dim * torch.log(2 * PI)

        return ll

    def _em_mle(self, values, stats, normalize):
        raise NotImplementedError("Maximum likelihood estimators not implemented for momentum models. Use autograd.")

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
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)

        optimizer = optimizers[optimizer](
            [diffusion, decay],
            lr=lr
        )

        prev_loss = np.inf

        Ezz  = stats.Ezz
        Ezz1 = stats.Ezz1

        for epoch in range(n_epochs):
            def loss_closure():
                optimizer.zero_grad()

                M = -torch.exp(decay) * self.dt + 1 
                sigma = torch.exp(diffusion)**2 * self.dt

                total_loss = 0
                for i in range(len(values.observations)):
                    loss = 0. 
                    T = values.observations[i].shape[0]


                    loss = Ezz[i][2:] - 2 * M * Ezz1[i][1:] + M **2 * Ezz[i][1:-1]
                    loss = torch.sum(loss,axis=0) / sigma
                    loss = (T-2) * torch.log(sigma**2) + loss

                    total_loss += loss / 2
                total_loss.backward(retain_graph=True)
                return total_loss

            total_loss = optimizer.step(loss_closure)

            if epoch > 0 and abs((total_loss.item() - prev_loss) / prev_loss) < gd_tol: 
                break

            prev_loss = total_loss.item()
        
        self.decay             = decay.detach()
        self.diffusion         = diffusion.detach()

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return torch.tensor(prev_loss)

    def fit(self, 
            X=None, 
            n_iter: int = 1000, 
            emtol: float = 1e-3, 
            maximization_type: str = 'autograd', 
            checkpoint_path: str|None = None,
            **diff_args
        ) -> MomentumResults:
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            X (None): Value ignored. We treat self.approx_mean as the observed variable.
            n_iter (int): Number of EM iterations.
            emtol (float): Tolerance for the change in log-likelihood between iterations.
            maximization_type (str): Type of maximization algorithm to use. In this model, only 'autograd' is implemented.
            checkpoint_path (str|None): Path to save checkpoint files. Checkpoint files are deleted after a successful run.
            **diff_args: Keyword arguments to pass to the parent class's maximization method.

        Returns:
            MomentumResults: Results of fitting the model to the data.
        """
        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        values = self._initialize(self.approximate_mean)

        if checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)

        for i in range(n_iter):
            with torch.no_grad():
                values = self.filter(values)
                values = self.smooth(values)
            ll = self._em(
                values,
                normalize=False,
                maximization_type=maximization_type,
                **diff_args
            )

            values.loglike.append(-ll)
            if not torch.isfinite(values.loglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.loglike[-1] - values.loglike[-2]) / values.loglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

            if i % 50 == 0:
                print(f"Iteration {i}: {-ll.item()}")
                if checkpoint_path:
                    hseu.save_pickle(values, f"./{checkpoint_path}/momentum_epoch_{i}.pkl")
                    hseu.save_pickle(self, f"./{checkpoint_path}/momentum_model_epoch_{i}.pkl")

        
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        values.cumulative_probabilities = self._calculate_marginals(self.environment_size, self.bin_size, values)

        return values