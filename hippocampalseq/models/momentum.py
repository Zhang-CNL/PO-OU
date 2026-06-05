import os
import numpy as np
import numpy.typing as npt
import pynapple as nap
import torch
import torch.nn as nn
import warnings
from typing import List, Tuple
from dataclasses import dataclass, field
from scipy.optimize import least_squares
from torch.utils.data import random_split

import hippocampalseq.utils as hseu

from .kalman_filter import *
__all__ = [
    'Momentum',
    'MomentumResults'
]

def resume_from_checkpoint(
        checkpoint_path: str, 
        n_iter: int = 100, 
        emtol: float = 1e-3, 
        maximization_type: str = 'autograd', 
        normalize: bool = False, 
        **diff_args
    ) -> Tuple[Momentum, MomentumResults]:
    nums = []
    for path in os.listdir(checkpoint_path):
        if path.endswith(".pkl"):
            nums.append(int(path.split("_")[2].split(".")[0]))
    nums.sort()
    last_iter = nums[-1]
    model = hseu.load_pickle(f"./{checkpoint_path}/momentum_epoch_{last_iter}.pkl")
    values = model._resume_from_checkpoint(
        last_iter,
        None,
        n_iter,
        emtol,
        maximization_type,
        normalize,
        **diff_args
    )
    return model,values
    

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
            adjust_parameters: bool = False,
            approximation_method: str = 'analytic',
            seed: int|None = 42
        ):
        """Initialize the momentum SSM.
        
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (tuple): Size of the environment. (xmin, ymin, xmax, ymax)
            bins (int): Number of bins for each latent dimension.
            adjust_parameters (bool): Whether to adjust the parameters. Not sure why we would do this, but it comes from Krause & Drugowitsch.
            seed: (int|None): Seed for the random number generator
        """
        super().__init__(2, 2, 2)

        assert approximation_method in ['analytic', 'iterative']
        self.__SCALING_FACTOR = torch.tensor(1000)

        self.dt               = torch.tensor(dt)
        self.environment_size = environment_size
        self.bin_size         = bin_size
        assert len(environment_size) == 2*self.latent_dim, "Environment shape and latent dimensions must match"


        x = torch.arange(environment_size[0], environment_size[2], bin_size) + bin_size/2
        y = torch.arange(environment_size[1], environment_size[3], bin_size) + bin_size/2
        self.grid = hseu.bin_points(x,y)

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
        #self.entropy                = []
        for k,v in enumerate(spikemat):
            ep = torch.from_numpy(v).double()
            emission_probability = hseu.calc_poisson_emission_probabilities_2d(
                ep, 
                place_fields,
                self.dt
            )
            emission_probability /= torch.sum(emission_probability, axis=(1,2), keepdim=True)

            if approximation_method == 'iterative':
                T = emission_probability.shape[0]
                approx_mean = torch.zeros(T, self.latent_dim, 1)
                approx_cov  = torch.zeros(T, self.latent_dim, self.latent_dim)
                #entropy     = torch.zeros(T)
                for t in range(T):
                    approx_mean[t], approx_cov[t],_ = hseu.iterative_gaussian_approximation(
                        self.grid,
                        emission_probability[t],
                        self.bin_size
                    )
            elif approximation_method == 'analytic':
                approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
                    self.grid,
                    emission_probability,
                    self.bin_size
                )
            
            self.emission_probabilities.append(emission_probability)
            self.approximate_mean.append(approx_mean)
            self.approximate_covariance.append(approx_cov)
            #self.entropy.append(entropy)


        # Random initialization of parameters
        self.decay             = torch.rand(1) 
        self.diffusion         = torch.rand(1) 
        self.initial_diffusion = torch.rand(1) 
        if adjust_parameters:
            with torch.no_grad():
                self.decay, self.diffusion = self._adjust_parameters(
                    self.decay, 
                    self.diffusion, 
                    self.dt
                )

        # TODO: Simple Bayesian decoder to get z from my approx_mean
        #a,b = _init_momentum_params(self.approx_mean.numpy())
        #print(a,b)
        # TODO:
        # Instead of randomly initializing parameters,
        # fit a plane to parameters based on P(z_t|z_{t-1},z_{t-2})
        # Use approx_mean as z


    def _adjust_parameters(self, theta, sigma, dt):
        n = 10**10
        t_adjusted = torch.log(dt * theta + 1) / dt 
        delta = n * dt 
        cfunction = (
            sigma ** 2 / theta * (
                (2 * theta * delta) - torch.exp(-2 * theta * delta)
                + 4 * torch.exp(-theta * delta)
                -3
            ) / (2 * theta**2)
        )
        prefactor = dt ** 2 / (2 * t_adjusted)
        numer = (
            (delta / dt) * -torch.exp(2 * t_adjusted * dt)
            - 2 * torch.exp(-t_adjusted * (delta - dt))
            - 2 * torch.exp(-t_adjusted * delta)
            + torch.exp(-2 * t_adjusted * delta)
            + 2 * torch.exp(t_adjusted * dt)
            + (delta / dt)
            + 1
        )
        denom = (torch.exp(t_adjusted * dt) - 1) ** 2 
        dfunction = prefactor * -(numer / denom)
        sigma_adjusted = torch.sqrt(cfunction / dfunction)
        return t_adjusted, sigma_adjusted

    def _initialize(self, X: torch.Tensor) -> MomentumResults:
        tf_base = [torch.zeros((x.shape[0], self.augmented_dim, 1)) for x in X]
        cov_base = [torch.zeros((x.shape[0], self.augmented_dim, self.augmented_dim)) for x in X]
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

    def _construct_init_var(self, initial_diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        init_cov = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_cov[:self.latent_dim, :self.latent_dim] = initial_diffusion * initial_diffusion \
                                                        * self.dt * I \
                                                        * self.__SCALING_FACTOR**2
        init_cov[self.latent_dim:, self.latent_dim:] = jitter * I
        return init_cov

    def _construct_transition_mat(self, decay: torch.Tensor, diffusion: torch.Tensor) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        ex     = torch.exp(-decay * self.__SCALING_FACTOR * self.dt)
        A1     = I * (1 + ex)
        A2     = I * ex
        top    = torch.cat((A1, A2), dim=1)
        bottom = torch.cat((I, Z), dim=1)
        A = torch.cat((top, bottom), dim=0)
        return A

    def _construct_transition_cov(self, decay: torch.Tensor, diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        Q = self.__SCALING_FACTOR * (diffusion * self.dt) ** 2 / (2*decay) * (1 - torch.exp(-2*decay * self.__SCALING_FACTOR * self.dt)) * I
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
        I = torch.eye(self.latent_dim)
        # $z_2 = I z_1 + \sigma_0^2dt\xi_1$
        init_mean = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_mean[:self.latent_dim, :self.latent_dim] = I
        init_mean[self.latent_dim:, :self.latent_dim] = I
        init_cov = self._construct_init_var(self.initial_diffusion)
        return init_mean, init_cov

    def _init_transition_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(self.decay, self.diffusion)
        Q = self._construct_transition_cov(self.decay, self.diffusion) 

        return A,Q

    def _init_observation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        C = torch.hstack((I, Z))
        H = self.approximate_covariance
        return C,H

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
        I = torch.eye(self.augmented_dim)
        # Deal with the uniform prior using the information filter
        # Since the uniform prior contains 0 precision, only the likelihood function contributes 
        sigma0i = torch.inverse(self.observation_covariance[batch][0])
        omega0 = self.observation_matrices.T @ sigma0i @ self.observation_matrices
        xi0 = self.observation_matrices.T @ sigma0i @ values.observations[batch][0]

        P0  = torch.inverse(omega0 + .00001 * torch.eye(self.augmented_dim))
        #P0 = torch.inverse(omega0)
        mu0 = P0 @ xi0
        P0  = 0.5 * (P0 + P0.T)

        # Filtered mean and covariance are augmented state spaces
        # $s_t = (z_t, z_{t-1})^T$
        values.filtered_mean[batch][0]  = mu0
        values.filtered_cov[batch][0]   = P0
        values.predicted_mean[batch][0] = self.initial_state_mean @ mu0
        values.predicted_cov[batch][0]  = self.initial_state_mean @ P0 @ self.initial_state_mean.T + self.initial_state_covariance

        # Now we can calculate it for $P(z_1|z_0)$
        return values

    def _filter(self, values: MomentumResults, batch: int, t: int) -> MomentumResults:
        """Run the Kalman filter for a single time step.
        Use our initial transition and covariance matrices for t == 0

        Args:
            values (MomentumResults): Results from previous filter passes.
            batch (int): Batch index.
            t (int): Time index.

        Returns:
            MomentumResults: Filtered results
        """
        
        A = self.transition_matrices if t > 1 else self.initial_state_mean
        C = self.observation_matrices
        gamma = self.transition_covariance if t > 1 else self.initial_state_covariance
        sigma = self.observation_covariance[batch]

        Am1 = values.predicted_mean[batch][t-1]
        Pn1 = values.predicted_cov[batch][t-1]

        PnCt = Pn1 @ C.T
        K = hseu.invmul(PnCt, C @ PnCt + sigma[t])

        mu_t = Am1 + K @ (values.observations[batch][t] - C @ Am1)
        v_t = (torch.eye(self.augmented_dim) - K @ C) @ Pn1

        Am = A @ mu_t
        Pt = A @ v_t @ A.T + gamma
        #Pt = .5 * (Pt + Pt.T)

        values.predicted_mean[batch][t] = Am
        values.predicted_cov[batch][t]  = Pt
        values.filtered_mean[batch][t]  = mu_t
        values.filtered_cov[batch][t]   = v_t

        return values

    def _smooth(self, values: MomentumResults, batch: int, t: int) -> MomentumResults:
        """Smooth the Kalman filter results for one timestep.

        If t == 0, use the initial state mean and covariance to calculate the smoothed values.
        Otherwise, use the previous smoothed values and the transition matrices to calculate the current smoothed values.

        Args:
            values (MomentumResults): Results from previous filter and smoothing passes.
            batch (int): Batch index.
            t (int): Time index.

        Returns:
            MomentumResults: Smoothed results.
        """
        if t == 0:
            Amt = values.predicted_mean[batch][t]
            Pt  = values.predicted_cov[batch][t]

            #J = hseu.invmul(values.filtered_cov[batch][t] @ self.initial_state_mean.T , Pt + .00001 * torch.eye(self.augmented_dim))
            J = hseu.invmul(values.filtered_cov[batch][t] @ self.initial_state_mean.T , Pt)
            muht = values.filtered_mean[batch][t] + J @ (values.smoothed_mean[batch][t+1] - Amt) 
            vht = values.filtered_cov[batch][t] + J @ (values.smoothed_cov[batch][t+1] - Pt) @ J.mT

            values.smoothed_gain[batch][t] = J
            values.smoothed_mean[batch][t] = muht
            values.smoothed_cov[batch][t]  = vht
            return values
        else:
            return super()._smooth(values, batch, t)

    def _loglikelihood(self, values: MomentumResults, stats: KalmanStatistics) -> torch.Tensor:
        """Calculate the log-likelihood for the model given the current state.
        
        Args:
            values (MomentumResults): The current decoded values for hidden states.
            stats (KalmanStatistics): Sufficient statistics calculated from these values.

        Returns:
            torch.Tensor: The log-likelihood for the model.
        """
        ll = 0
        A = self.transition_matrices
        C = self.observation_matrices
        Gamma = self._construct_transition_cov(self.decay, self.diffusion, .00001)#self.transition_covariance
        Sigma = self.observation_covariance

        init_mat = self.initial_state_mean
        init_cov = self._construct_init_var(self.initial_diffusion, .00001)#self.initial_state_covariance

        ll = 0
        for b in range(len(values.observations)):
            T = values.observations[b].shape[0]

            _ll = 0
            _ll += torch.logdet(init_cov)
            _ll += torch.logdet(Gamma) * (T-2)
            _ll += torch.sum(torch.logdet(Sigma[b]))

            ill = stats.Ezz[b][1] - stats.Ezz1[b][0] @ init_mat.mT - init_mat @ stats.Ez1z[b][0] + init_mat @ stats.Ezz[b][0] @ init_mat.mT
            ill = torch.linalg.solve(init_cov, ill)
            _ll += torch.trace(ill) 

            tll = stats.Ezz[b][2:] - stats.Ezz1[b][1:] @ A.mT - A @ stats.Ez1z[b][1:]  + A @ stats.Ezz[b][1:-1] @ A.mT
            tll = torch.sum(tll, axis=0)
            tll = torch.linalg.solve(Gamma, tll)
            _ll += torch.trace(tll) 

            ell = stats.Exx[b] - stats.Exz[b] @ C.mT - C @ stats.Ezx[b] + C @ stats.Ezz[b] @ C.mT
            ell = torch.linalg.solve(Sigma[b] + .00001 * torch.eye(self.obs_dim), ell)
            ell = torch.sum(ell, axis=0)
            _ll += torch.trace(ell) 

            _ll += T * self.augmented_dim * torch.log(2 * PI)
            ll += _ll / 2

        return ll

    def _em_mle(self, values, stats, normalize):
        raise NotImplementedError("Maximum likelihood estimators not implemented for momentum models. Use autograd.")

    def _em_autograd(self, 
            values: MomentumResults,
            stats: SufficientStatistics,
            normalize: bool,
            lr: float = 1e-3, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
            seed: int|None = 42
        ) -> torch.Tensor:
        """Perform maximum likelihood estimation of all relevant parameters for the momentum SSM using autograd.

        Args:
            values (MomentumResults): Momentum filtering pass results.
            stats (SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
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

        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        decay             = torch.zeros(1, requires_grad=True)
        diffusion         = torch.zeros(1, requires_grad=True)
        initial_diffusion = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            initial_diffusion.copy_(self.initial_diffusion)

        optimizer = torch.optim.Adam(
            [diffusion, decay, initial_diffusion],
            lr=lr
        )

        jitter = torch.eye(self.obs_dim) * .000001
        prev_loss = np.inf

        for epoch in range(n_epochs):
            total_loss = 0

            for i in range(len(values.observations)):
                A = self._construct_transition_mat(decay, diffusion)
                C = self.observation_matrices
                Gamma = self._construct_transition_cov(decay, diffusion, jitter=1e-6)
                Sigma = self.observation_covariance[i]

                init_mat = self.initial_state_mean
                init_cov = self._construct_init_var(initial_diffusion, jitter=1e-6)

                loss = 0.
                T = values.observations[i].shape[0]

                ill = stats.Ezz[i][1] - stats.Ezz1[i][0] @ init_mat.mT - init_mat @ stats.Ez1z[i][0] + init_mat @ stats.Ezz[i][0] @ init_mat.mT
                ill = torch.linalg.solve(init_cov, ill)
                loss += torch.trace(ill)

                tll = stats.Ezz[i][2:] - stats.Ezz1[i][1:] @ A.mT - A @ stats.Ez1z[i][1:]  + A @ stats.Ezz[i][1:-1] @ A.mT
                tll = torch.sum(tll, axis=0)
                tll = torch.linalg.solve(Gamma, tll)
                loss += torch.trace(tll)

                ell = stats.Exx[i] - stats.Exz[i] @ C.mT - C @ stats.Ezx[i] + C @ stats.Ezz[i] @ C.mT
                ell = torch.linalg.solve(Sigma, ell)
                ell = torch.sum(ell, axis=0)
                loss += torch.trace(ell)

                loss += torch.sum(torch.logdet(Sigma))
                loss += torch.logdet(init_cov)
                loss += torch.logdet(Gamma) * (T - 2)
                loss /= 2.0 

                total_loss += loss / len(values.observations)
            
            if epoch > 0 and abs((total_loss.item() - prev_loss) / prev_loss) < gd_tol: 
                break

            total_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

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

    def _resume_from_checkpoint(self, 
            last_iter: int,
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'autograd', 
            normalize: bool = False, 
            checkpoint_path: str|None = "./checkpoint/",
            **diff_args
        ) -> MomentumResults:
        """Resume training from a checkpoint. 
        Checkpoint files have the pattern `momentum_epoch_{i}.pkl` where `i` is the epoch number.

        Args:
            last_iter (int): The last epoch number from which to resume training.
            n_iter (int, optional): The number of epochs to train for. Defaults to 100.
            emtol (float, optional): The tolerance for the EM algorithm. Defaults to 1e-3.
            maximization_type (str, optional): The type of maximization to use. Defaults to 'autograd'.
            normalize (bool, optional): Whether to normalize the transition and observation matrices. Defaults to False.
            checkpoint_path (str, optional): The path to save checkpoint files to. Defaults to "./checkpoint/".
            **diff_args: Keyword arguments to pass to the `_em` method.

        Returns:
            MomentumResults: The results of the EM algorithm. Estimated parameters can be accessed from this class itself.
        """
        values = self._initialize(self.approximate_mean)


        for i in range(last_iter + 1, n_iter):
            with torch.no_grad():
                values = self.filter(values)
                values = self.smooth(values)
            ll = self._em(
                values,
                normalize,
                maximization_type,
                **diff_args
            )

            values.negloglike.append(-ll)
            if not torch.isfinite(values.negloglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.negloglike[-1] - values.negloglike[-2]) / values.negloglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

            if i % 50 == 0:
                print(f"Iteration {i}: {-ll.item()}")
                if checkpoint_path:
                    hseu.save_pickle(values, f"./{checkpoint_path}/momentum_epoch_{i}.pkl")
        
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        values.cumulative_probabilities = self._calculate_marginals(values)

        os.rmdir(checkpoint_path)

        return values

    def fit(self, 
            X=None, 
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'autograd', 
            normalize: bool = False, 
            checkpoint_path: str|None = "./checkpoint/",
            **diff_args
        ) -> MomentumResults:
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            X (None): Value ignored. We treat self.approx_mean as the observed variable.
            n_iter (int): Number of EM iterations.
            emtol (float): Tolerance for the change in log-likelihood between iterations.
            maximization_type (str): Type of maximization algorithm to use. In this model, only 'autograd' is implemented.
            normalize (bool): Whether to normalize the transition and observation matrices. Defaults to False.
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
                normalize,
                maximization_type,
                **diff_args
            )

            values.negloglike.append(-ll)
            if not torch.isfinite(values.negloglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.negloglike[-1] - values.negloglike[-2]) / values.negloglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

            if i % 50 == 0:
                print(f"Iteration {i}: {-ll.item()}")
                if checkpoint_path:
                    hseu.save_pickle(values, f"./{checkpoint_path}/momentum_epoch_{i}.pkl")

        
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        values.cumulative_probabilities = self._calculate_marginals(values)

        return values