import os
import jax 
jax.config.update("jax_enable_x64", True)

import optax
import numpy as np
import numpy.typing as npt
import pynapple as nap
import jax.numpy as jnp
import warnings
from typing import List, Tuple
from dataclasses import dataclass, field
from scipy.optimize import least_squares

import hippocampalseq.utils as hseu

from .kalman_filter import *
from .statespace import *

__all__ = [
    'Momentum',
    'MomentumResults'
]
@jax.jit
def loss_closure(params, dt, Ezz, Ezz1):
    decay,diffusion,initial_diffusion = params
    _idiff = jnp.exp(initial_diffusion)
    _diff  = jnp.exp(diffusion)
    _decay = jnp.exp(decay)

    v1 = _idiff**2 * dt
    alpha = 1 + jnp.exp(-_decay * dt)
    gamma = (_diff * dt)**2 / (2 * _decay) * (1 - jnp.exp(-2 * _decay * dt))
    total_loss = 0
    for i in range(len(Ezz)):
        loss = 0. 
        T = Ezz[i].shape[0]

        ill = Ezz[i][1] - 2 * Ezz1[i][0] + Ezz[i][0]
        ill = ill / v1
        ill = jnp.log(v1**2) + ill
        loss += ill

        tll = Ezz[i][2:] - 2 * alpha * Ezz1[i][1:] + alpha**2 * Ezz[i][1:-1]
        tll = jnp.sum(tll,axis=0) / gamma
        tll = (T-2) * jnp.log(gamma**2) + tll
        loss += tll

        total_loss += loss / 2
    return total_loss.squeeze()

def resume_from_checkpoint(
        checkpoint_path: str, 
        n_iter: int = 100, 
        emtol: float = 1e-3, 
        maximization_type: str = 'autograd', 
        **diff_args
    ): #-> Tuple[Momentum, MomentumResults]:
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
        **diff_args
    )
    return model,values
    

@dataclass
class MomentumResults(KalmanResults):
    emission_probabilities : List[jax.Array] = field(default_factory=list)
    approximate_mean       : List[jax.Array] = field(default_factory=list)
    approximate_covariance : List[jax.Array] = field(default_factory=list)

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
            seed: int|None = 42
        ):
        """Initialize the momentum SSM.
        
        Args:
            place_fields (np.ndarray|jax.Array): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|jax.Array): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (tuple): Size of the environment. (xmin, ymin, xmax, ymax)
            bin_size (int): Size of individual bins in cm.
            data_type (str): Type of data. Either 'replay' or 'theta'.
            adjust_parameters (bool): Whether to adjust the parameters. Not sure why we would do this, but it comes from Krause & Drugowitsch.
            seed: (int|None): Seed for the random number generator
        """
        super().__init__(2, 2, 2, seed=seed)

        self.dt               = dt
        self.environment_size = environment_size
        self.bin_size         = bin_size
        assert len(environment_size) == 2*self.latent_dim, "Environment shape and latent dimensions must match"

        self.grid = hseu.create_grid(self.environment_size, self.bin_size)
        place_fields = jnp.array(place_fields, dtype=jnp.float64)

        values = MomentumResults(
            emission_probabilities = [],
            approximate_mean       = [],
            approximate_covariance = []
        )
        self.emission_probabilities = []
        self.approximate_mean       = []
        self.approximate_covariance = []
        for k,v in enumerate(spikemat):
            ep = jnp.array(v, dtype=jnp.float64)
            emission_probability = hseu.calc_poisson_emission_probabilities_2d(
                ep, 
                place_fields,
                self.dt
            )
            emission_probability /= jnp.sum(emission_probability, axis=(1,2), keepdims=True)

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
        self.decay             = self.random(1)
        self.diffusion         = self.random(1)
        self.initial_diffusion = self.random(1) 
        if adjust_parameters:
            self.decay, self.diffusion = self._adjust_parameters(
                self.decay, 
                self.diffusion, 
                self.dt
            )

        self.n_parameters = 3

        # TODO: Simple Bayesian decoder to get z from my approx_mean
        #a,b = _init_momentum_params(self.approx_mean.numpy())
        #print(a,b)
        # TODO:
        # Instead of randomly initializing parameters,
        # fit a plane to parameters based on P(z_t|z_{t-1},z_{t-2})
        # Use approx_mean as z

    def _adjust_parameters(self, theta, sigma, dt):
        # TODO: Make this in log form
        n = 10**10
        t_adjusted = jnp.log(dt * theta + 1) / dt 
        delta = n * dt 
        cfunction = (
            sigma ** 2 / theta * (
                (2 * theta * delta) - jnp.exp(-2 * theta * delta)
                + 4 * jnp.exp(-theta * delta)
                -3
            ) / (2 * theta**2)
        )
        prefactor = dt ** 2 / (2 * t_adjusted)
        numer = (
            (delta / dt) * -jnp.exp(2 * t_adjusted * dt)
            - 2 * jnp.exp(-t_adjusted * (delta - dt))
            - 2 * jnp.exp(-t_adjusted * delta)
            + jnp.exp(-2 * t_adjusted * delta)
            + 2 * jnp.exp(t_adjusted * dt)
            + (delta / dt)
            + 1
        )
        denom = (jnp.exp(t_adjusted * dt) - 1) ** 2 
        dfunction = prefactor * -(numer / denom)
        sigma_adjusted = jnp.sqrt(cfunction / dfunction)
        return t_adjusted, sigma_adjusted

    def _initialize(self, X: jax.Array) -> MomentumResults:
        tf_base = [np.ones((x.shape[0], self.augmented_dim, 1)) for x in X]
        cov_base = [np.ones((x.shape[0], self.augmented_dim, self.augmented_dim)) for x in X]
        def _copy(x):
            return [i.copy() for i in x]
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

    def _construct_init_mean(self) -> jax.Array:
        I = jnp.eye(self.latent_dim, dtype=jnp.float64)
        Z = jnp.zeros((self.augmented_dim, self.latent_dim), dtype=jnp.float64)
        left = jnp.concat((I, I), axis=0)
        return jnp.concat((left, Z), axis=1)

    def _construct_init_var(self, initial_diffusion: jax.Array, jitter=0.0) -> jax.Array:
        I = jnp.eye(self.latent_dim, dtype=jnp.float64)
        Z = jnp.zeros((self.latent_dim,self.latent_dim), dtype=jnp.float64)
        idiff = initial_diffusion**2 * self.dt * I
        top = jnp.concat((idiff, Z), axis=1)
        bottom = jnp.concat((Z, I * jitter), axis=1)
        return jnp.concat((top, bottom), axis=0)

    def _construct_transition_mat(self, decay: jax.Array) -> jax.Array:
        I = jnp.eye(self.latent_dim, dtype=jnp.float64)
        Z = jnp.zeros((self.latent_dim, self.latent_dim), dtype=jnp.float64)

        ex     = jnp.exp(-decay * self.dt)
        A1     = I * (1 + ex)
        A2     = I * ex
        top    = jnp.concat((A1, A2), axis=1)
        bottom = jnp.concat((I, Z), axis=1)
        A = jnp.concat((top, bottom), axis=0)
        return A

    def _construct_transition_cov(self, decay: jax.Array, diffusion: jax.Array, jitter=0.0) -> jax.Array:
        I = jnp.eye(self.latent_dim, dtype=jnp.float64)
        Z = jnp.zeros((self.latent_dim, self.latent_dim), dtype=jnp.float64)

        Q = (diffusion * self.dt) ** 2 / (2*decay) \
            * (1 - jnp.exp(-2*decay * self.dt)) * I
        top    = jnp.concat((Q, Z), axis=1)
        bottom = jnp.concat((Z, I * jitter), axis=1)
        Gamma = jnp.concat((top, bottom), axis=0)
        return Gamma

    def _init_priors(self) -> Tuple[jax.Array, jax.Array]:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        Meanwhile, $P(z_2|z_1) = \mathcal{N}(z_2|z_1, \sigma_0^2 dt)$: a simple gaussian.

        Returns:
            (jax.Array): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (jax.Array): Prior covariance for augmented state
        """
        # $z_2 = I z_1 + \sigma_0^2dt\xi_1$
        init_mean = self._construct_init_mean()
        init_cov = self._construct_init_var(jnp.exp(self.initial_diffusion))
        return init_mean, init_cov

    def _init_transition_matrices(self) -> Tuple[jax.Array, jax.Array]:
        """Construct transition matrices for momentum SSM.

        Returns:
            (jax.Array): transition matrix for augmented state $[z_t; z_{t-1}]^T$
            (jax.Array): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(jnp.exp(self.decay))
        Q = self._construct_transition_cov(jnp.exp(self.decay), jnp.exp(self.diffusion)) 
        return A,Q

    def _init_observation_matrices(self) -> Tuple[jax.Array, jax.Array]:
        I = jnp.eye(self.latent_dim, dtype=jnp.float64)
        Z = jnp.zeros((self.latent_dim, self.latent_dim), dtype=jnp.float64)
        H = jnp.hstack((I, Z))
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
        #"""
        mu1 = values.observations[batch][0]
        P1  = self.observation_covariance[batch][0]

        """
        dx = self.environment_size[2] - self.environment_size[0]
        dy = self.environment_size[3] - self.environment_size[1]
        mu0 = torch.tensor([
            [dx / 2],
            [dy / 2]
        ])
        P0 = torch.tensor([
            [dx**2 / 12, 0],
            [0, dy**2 / 12]
        ])
        # C is an identity matrix here, so we can ignore it.
        K1  = hseu.invmul(P0, P0 + self.observation_covariance[batch][0])
        mu1 = mu0 + K1 @ (values.observations[batch][0] - mu0)
        P1  = (torch.eye(self.latent_dim) - K1) @ P0
        """

        # Filtered mean and covariance are augmented state spaces
        # $s_t = (z_t, z_{t-1})^T$
        """
        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = P1
        """
        #values.filtered_mean[batch][0]  = mu1.repeat(self.latent_dim,1)
        #values.filtered_cov[batch][0]   = P1.repeat(self.latent_dim,self.latent_dim)
        values.filtered_mean[batch][0]  = jnp.vstack((mu1,)*self.latent_dim)
        values.filtered_cov[batch][0]   = jnp.tile(P1, (self.latent_dim, self.latent_dim))
        #"""
        values.predicted_mean[batch][0] = self.initial_state_mean @ values.filtered_mean[batch][0]
        values.predicted_cov[batch][0]  = self.initial_state_mean @ values.filtered_cov[batch][0] @ self.initial_state_mean.T + self.initial_state_covariance

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
        sigma = self.observation_covariance[batch][t]

        Am1 = values.predicted_mean[batch][t-1]
        Pn1 = values.predicted_cov[batch][t-1]

        PnCt = Pn1 @ C.T
        K    = hseu.invmul(PnCt, C @ PnCt + sigma)

        mu_t = Am1 + K @ (values.observations[batch][t] - C @ Am1)
        v_t  = (jnp.eye(self.augmented_dim) - K @ C) @ Pn1

        Am = self.transition_matrices @ mu_t
        Pt = self.transition_matrices @ v_t @ self.transition_matrices.T + self.transition_covariance

        values.predicted_mean[batch][t] = Am   # $\mu_{t+1|t}$
        values.predicted_cov[batch][t]  = Pt   # $P_{t+1|t}$
        values.filtered_mean[batch][t]  = mu_t # $\mu_{t|t}$
        values.filtered_cov[batch][t]   = v_t  # $P_{t|t}$

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
            J    = hseu.invmul(values.filtered_cov[batch][t] @ self.initial_state_mean.T , Pt)
            muht = values.filtered_mean[batch][t] + J @ (values.smoothed_mean[batch][t+1] - Amt) 
            vht  = values.filtered_cov[batch][t] + J @ (values.smoothed_cov[batch][t+1] - Pt) @ J.mT

            values.smoothed_gain[batch][t] = J
            values.smoothed_mean[batch][t] = muht
            values.smoothed_cov[batch][t]  = vht
            return values
        else:
            return super()._smooth(values, batch, t)

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
            jnp.trace(c, axis1=-2, axis2=-1).squeeze() + (sm[1:,:self.latent_dim].mT @ sm[:-1,:self.latent_dim]).squeeze()
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

    def _loglikelihood(self, values: MomentumResults, stats: KalmanStatistics) -> jax.Array:
        """Calculate the log-likelihood for the model given the current state.
        
        Args:
            values (MomentumResults): The current decoded values for hidden states.
            stats (KalmanStatistics): Sufficient statistics calculated from these values.

        Returns:
            jax.Array: The log-likelihood for the model.
        """
        #ell = stats.Exx[b] - stats.Exz[b] @ C.mT - C @ stats.Ezx[b] + C @ stats.Ezz[b] @ C.mT
        #ell = torch.linalg.solve(Sigma[b] + .00001 * torch.eye(self.obs_dim), ell)
        #ell = torch.sum(ell, axis=0)
        #_ll += torch.trace(ell) 

        ll = 0
        idiff = jnp.exp(self.initial_diffusion)
        diff  = jnp.exp(self.diffusion)
        decay = jnp.exp(self.decay)

        Sigma = self.observation_covariance
        v1 = idiff**2 * self.dt 
        alpha = 1 + jnp.exp(-decay * self.dt)
        gamma = (diff * self.dt)**2 / (2*decay) * (1 - jnp.exp(-2*decay * self.dt))

        for b in range(len(values.observations)): 
            T = values.observations[b].shape[0]
            _ll = 0
            _ll += jnp.log(v1**2)
            _ll += jnp.log(gamma**2) * (T-2)
            #_ll += jnp.sum(jnp.logdet(Sigma[b]))

            ill = stats.Ezz[b][1] - 2*stats.Ezz1[b][0] + stats.Ezz[b][0]
            ill = ill / v1 
            _ll += ill

            tll = stats.Ezz[b][2:] - 2*alpha*Ezz1[b][1:] + alpha**2 * stats.Ezz[b][1:-1]
            tll = jnp.sum(tll, axis=0) / gamma 
            _ll += tll 

            #ell = stats.Exx[b] - stats.Ezx[b] - stats.Ezx[b].mT + stats.Ezz[b]

            ll += _ll / 2 + T * self.augmented_dim * jnp.log(2 * np.pi)

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
        ) -> jax.Array:
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
            jax.Array: The final negative log likelihood.
        """

        decay             = self.decay.copy()
        diffusion         = self.diffusion.copy()
        initial_diffusion = self.initial_diffusion.copy()
        params = jnp.hstack([decay, diffusion, initial_diffusion])

        params,prev_loss = self._fit_ag(
            loss_closure,
            params,
            optimizer,
            lr,
            n_epochs,
            gd_tol,
            self.dt,
            stats.Ezz,
            stats.Ezz1
        )
        
        self.decay,self.diffusion,self.initial_diffusion = params

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return prev_loss

    def _resume_from_checkpoint(self, 
            last_iter: int,
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'autograd', 
            checkpoint_path: str|None = None,
            **diff_args
        ) -> MomentumResults:
        """Resume training from a checkpoint. 
        Checkpoint files have the pattern `momentum_epoch_{i}.pkl` where `i` is the epoch number.

        Args:
            last_iter (int): The last epoch number from which to resume training.
            n_iter (int, optional): The number of epochs to train for. Defaults to 100.
            emtol (float, optional): The tolerance for the EM algorithm. Defaults to 1e-3.
            maximization_type (str, optional): The type of maximization to use. Defaults to 'autograd'.
            checkpoint_path (str, optional): The path to save checkpoint files to. If `None`, no checkpoint files will be saved. Defaults to None.
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
                False,
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
            values = self.filter(values)
            values = self.smooth(values)
            #print("Filtered and smoothed")
            #raise Exception()
            ll = self._em(
                values,
                normalize=False,
                maximization_type=maximization_type,
                **diff_args
            )

            values.loglike.append(-ll)
            if not jnp.isfinite(values.loglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.loglike[-1] - values.loglike[-2]) / values.loglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

            if i % 50 == 0:
                print(f"Iteration {i}: {-ll}")
                if checkpoint_path:
                    hseu.save_pickle(values, f"./{checkpoint_path}/momentum_epoch_{i}.pkl")

        
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        values.cumulative_probabilities = self._calculate_marginals(self.environment_size, self.bin_size, values)

        return values