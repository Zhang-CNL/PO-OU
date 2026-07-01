import os
import numpy as np
import jax 
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hippocampalseq.utils as hseu
from .statespace import *

__all__ = [
    'KalmanFilter',
    'KalmanResults',
    'KalmanStatistics',
    'PI'
]

PI = np.pi

@dataclass
class KalmanResults:
    observations   : List[np.ndarray] = field(default_factory=list)
    predicted_mean : List[np.ndarray] = field(default_factory=list)
    predicted_cov  : List[np.ndarray] = field(default_factory=list)
    filtered_mean  : List[np.ndarray] = field(default_factory=list)
    filtered_cov   : List[np.ndarray] = field(default_factory=list)
    smoothed_gain  : List[np.ndarray] = field(default_factory=list)
    smoothed_mean  : List[np.ndarray] = field(default_factory=list)
    smoothed_cov   : List[np.ndarray] = field(default_factory=list)
    loglike        : List[float]      = field(default_factory=list)
    cumulative_probabilities : np.ndarray = field(default_factory=lambda: np.empty(0))

@dataclass
class KalmanStatistics:
    Cov  : List[np.ndarray] # $\hat{V}_tJ_{t-1}$
    Ez   : List[np.ndarray] # $\mathbb{E}[z^T]$
    Ezz  : List[np.ndarray] # $\mathbb{E}[zz^T]$
    Ezz1 : List[np.ndarray] # $\mathbb{E}[z_{t}z_{t-1}^T]$
    Ez1z : List[np.ndarray] # $\mathbb{E}[z_{t-1}z_t^T]$
    Exx  : List[np.ndarray] # $\mathbb{E}[xx^T]  $
    Exz  : List[np.ndarray] # $\mathbb{E}[xz^T]  $
    Ezx  : List[np.ndarray] # $\mathbb{E}[zx^T]$

class KalmanFilter(StateSpace):
    """Base state space modeling class that implements a kalman filter"""
    def __init__(self, latent_dim, obs_dim, order=1, seed: Optional[int] = None):
        """Initialize the kalman filter

        Args:
            latent_dim (int): Dimension of the latent state.
            obs_dim (int): Dimension of the observations.
            order (int, optional): Order of the state space model. The order of the markov chain being used. Defaults to 1.
        """
        if seed is None:
            seed = int.from_bytes(os.urandom(4))
        if isinstance(seed, int):
            seed = jax.random.key(seed)
        self.key = seed

        self.latent_dim    = latent_dim
        self.augmented_dim = order * latent_dim
        self.obs_dim       = obs_dim

        self.n_parameters = 3*(self.augmented_dim**2) \
            + self.augmented_dim \
            + self.obs_dim * self.augmented_dim \
            + self.obs_dim**2 
        # Transition matrix -> nxn
        # Transition noise -> nxn
        # Initial matrix -> nx1
        # Initial variance -> nxn
        # Observation matrix -> nxm
        # Observation noise -> mxm

    def _parse_observations(self, obs) -> List[jax.Array]:
        """Safely convert observations to their expected format."""
        if not isinstance(obs, list):
            obs = [obs]
        for i in range(len(obs)):
            obs[i] = np.atleast_3d(obs[i])
            if obs[i].shape[1] < obs[i].shape[2]:
                obs[i] = obs[i].mT
        return obs

    def _filter_init(self, values: KalmanResults, batch: int) -> KalmanResults:
        """Initialize the filter for the first observation."""
        P0Ct = self.initial_state_covariance @ self.observation_matrices.T 
        K1 = hseu.invmul(P0Ct, self.observation_matrices @ P0Ct + self.observation_covariance) 
        mu1 = self.initial_state_mean + K1 @ (values.observations[batch][0] - self.observation_matrices @ self.initial_state_mean)
        v1 = (jnp.eye(self.augmented_dim) - K1 @ self.observation_matrices) @ self.initial_state_covariance
        #mu1 = self.initial_state_mean
        #v1  = self.initial_state_covariance

        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = v1
        values.predicted_mean[batch][0] = self.transition_matrices @ mu1
        values.predicted_cov[batch][0]  = self.transition_matrices @ v1 @ self.transition_matrices.T + self.transition_covariance
        return values


    def _filter(self, values: KalmanResults, batch: int, t: int) -> KalmanResults:
        """Run the filter for a single time step."""
        Am1 = values.predicted_mean[batch][t-1]
        Pn1 = values.predicted_cov[batch][t-1]

        Pct = Pn1 @ self.observation_matrices.T
        K = hseu.invmul(Pct, self.observation_matrices @ Pct + self.observation_covariance)

        mut = Am1 + K @ (values.observations[batch][t] - self.observation_matrices @ Am1)
        vt  = (jnp.eye(self.augmented_dim) - K @ self.observation_matrices) @ Pn1

        Am = self.transition_matrices @ mut
        Pn = self.transition_matrices @ vt @ self.transition_matrices.T + self.transition_covariance

        values.filtered_mean[batch][t]  = mut
        values.filtered_cov[batch][t]   = vt
        values.predicted_mean[batch][t] = Am
        values.predicted_cov[batch][t]  = Pn
        return values

    def filter(self, values: KalmanResults) -> KalmanResults:
        """Run the filter."""
        # Initialize the filter
        for batch in range(len(values.observations)):
            values = self._filter_init(values, batch)
            for t in range(1, values.observations[batch].shape[0]):
                values = self._filter(values, jnp.array(batch), jnp.array(t))

        return values

    def _smooth_init(self, values: KalmanResults, batch: int) -> KalmanResults:
        """Initialize the RTS smoother."""
        values.smoothed_mean[batch][-1] = values.filtered_mean[batch][-1]
        values.smoothed_cov[batch][-1]  = values.filtered_cov[batch][-1]
        return values

    def _smooth(self, values: KalmanResults, batch: int, t: int) -> KalmanResults:
        """Run the RTS smoother for a single time step."""
        Amt = values.predicted_mean[batch][t]
        Pt  = values.predicted_cov[batch][t]

        J = hseu.invmul(values.filtered_cov[batch][t] @ self.transition_matrices.T, Pt)
        muht = values.filtered_mean[batch][t] + J @ (values.smoothed_mean[batch][t+1] - Amt) 
        vht = values.filtered_cov[batch][t] + J @ (values.smoothed_cov[batch][t+1] - Pt) @ J.mT

        values.smoothed_gain[batch][t] = J
        values.smoothed_mean[batch][t] = muht
        values.smoothed_cov[batch][t]  = vht

        return values

    def smooth(self, values: KalmanResults) -> KalmanResults:
        """Run the RTS smoother."""
        # Initialize the RTS smoother
        for batch in range(len(values.observations)):
            values = self._smooth_init(values, batch)
            for t in reversed(range(values.observations[batch].shape[0] - 1)):
                values = self._smooth(values, jnp.array(batch), jnp.array(t))
        return values

    def _init_priors(self) -> Tuple[jax.Array, jax.Array]:
        m = self.random((self.augmented_dim, 1), 'normal')
        n = jnp.eye(self.augmented_dim)
        return m,n

    def _init_transition_matrices(self) -> Tuple[jax.Array, jax.Array]:
        F = self.random((self.augmented_dim, self.augmented_dim))
        F = F / F.sum(axis=1, keepdims=True)
        Q = self.random((self.augmented_dim, self.augmented_dim), 'uniform')
        Q = Q @ Q.T 
        return F,Q

    def _init_observation_matrices(self) -> Tuple[jax.Array, jax.Array]:
        H = self.random((self.augmented_dim, self.obs_dim))
        H = H / H.sum(axis=1, keepdims=True)
        R = self.random((self.obs_dim, self.obs_dim))
        R = R @ R.T
        return H, R

    def _initialize_parameters(self) -> Tuple[jax.Array,...]:
        initial_mean, initial_cov = self._init_priors()
        trans_mat, trans_cov      = self._init_transition_matrices()
        obs_mat, obs_cov          = self._init_observation_matrices()
        return trans_mat, trans_cov, obs_mat, obs_cov, initial_mean, initial_cov

    def _initialize(self, X: jax.Array) -> KalmanResults:
        tf_base  = [np.zeros_like(x) for x in X]
        cov_base = [np.zeros(x.shape[:-1] + (self.augmented_dim,)) for x in X]
        def _copy(x):
            return [i.copy() for i in x]
        return KalmanResults(
            observations   = X,
            predicted_mean = _copy(tf_base),
            predicted_cov  = _copy(cov_base),
            filtered_mean  = _copy(tf_base),
            filtered_cov   = _copy(cov_base),
            smoothed_gain  = _copy(cov_base),
            smoothed_mean  = _copy(tf_base),
            smoothed_cov   = _copy(cov_base),
        )

    def _calc_sufficient_stats(self, values: KalmanResults) -> KalmanStatistics:
        """Calculate sufficient statistics for performing maximization given the filtered
         and smoothed values of the model.

        Args:
            values (KalmanResults): The filtered and smoothed values of the model.

        Returns:
            KalmanStatistics: The sufficient statistics of the model.
        """
        Cov  = [sc[1:] @ sg[:-1].mT for sc,sg in zip(values.smoothed_cov, values.smoothed_gain)]
        Ez   = values.smoothed_mean
        Ezz  = [sc + sm @ sm.mT for sc,sm in zip(values.smoothed_cov, values.smoothed_mean)]
        Ezz1 = [sm[1:] @ sm[:-1].mT + c for sm,c in zip(values.smoothed_mean, Cov)]
        Ez1z = [ez.mT for ez in Ezz1]
        Exx  = [obs @ obs.mT for obs in values.observations]
        Exz  = [obs @ sm.mT for obs,sm in zip(values.observations, values.smoothed_mean)]
        Ezx  = [ex.mT for ex in Exz]
        return KalmanStatistics(
            Cov, Ez, Ezz, Ezz1, Ez1z, Exx, Exz, Ezx
        )

    def _loglikelihood(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        """Calculate the log likelihood of the model given the sufficient statistics and current 
        parameters.

        Args:
            values (KalmanResults): The filtered and smoothed values of the model.
            stats (KalmanStatistics): The sufficient statistics of the model.

        Returns:
            jax.Array: The log likelihood of the model.
        """
        ll = 0
        for b in range(len(values.observations)):
            T = values.observations[b].shape[0]
            icd = hseu.logdet(self.initial_state_covariance)
            if jnp.isfinite(icd):
                ll += icd

            _ll = 0
            _ll += hseu.logdet(self.transition_covariance) * (T - 1)
            _ll += hseu.logdet(self.observation_covariance) * T

            ip1 = stats.Ezz[b][0]
            ip2 = self.initial_state_mean @ values.smoothed_mean[b][0].mT
            ip3 = values.smoothed_mean[b][0] @ self.initial_state_mean.mT
            ip4 = self.initial_state_mean @ self.initial_state_mean.mT
            ill = hseu.mulinv(self.initial_state_covariance + .001 * jnp.eye(self.augmented_dim), ip1 - ip2 - ip3 + ip4)
            ill = jnp.trace(ill, axis1=-2, axis2=-1) / 2
            _ll += ill

            tp1 = stats.Ezz[b][1:]
            tp2 = stats.Ezz1[b] @ self.transition_matrices.mT
            tp3 = tp2.mT 
            tp4 = self.transition_matrices @ stats.Ezz[b][:-1] @ self.transition_matrices.mT
            tll = jnp.sum(tp1 - tp2 - tp3 + tp4, axis=0) 
            tll = hseu.mulinv(self.transition_covariance, tll)
            tll = jnp.trace(tll, axis1=-2, axis2=-1) / 2
            _ll += tll

            ep1 = stats.Exx[b]
            ep2 = self.observation_matrices @ stats.Ezx[b]
            ep3 = ep2.mT
            ep4 = self.observation_matrices @ stats.Ezz[b] @ self.observation_matrices.mT
            ell = jnp.sum(ep1 - ep2 - ep3 + ep4, axis=0)
            ell = hseu.mulinv(self.observation_covariance, ell)
            ell = jnp.trace(ell, axis1=-2, axis2=-1) / 2
            _ll += ell

            _ll += T * self.latent_dim * jnp.log(2*np.pi)
            ll += _ll / 2
        return ll

    def _initial_mean_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        return jnp.atleast_2d(
            jnp.mean(
                jnp.concat([jnp.expand_dims(sm[0], 0) for sm in values.smoothed_mean]),
                axis=0
            )
        )
    
    def _initial_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        """
        mu1 = torch.cat([sm[0].unsqueeze(0) for sm in values.smoothed_mean])
        c1  = torch.cat([sc[0].unsqueeze(0) for sc in values.smoothed_cov])
        P1 = c1 + mu1 @ mu1.mT 
        P2 = self.initial_state_mean @ mu1.mT 
        P3 = P2.mT 
        P4 = self.initial_state_mean @ self.initial_state_mean.mT 
        return torch.atleast_2d(torch.mean(P1 - P2 - P3 + P4, axis=0))
        """ 
        P1 = jnp.concat([jnp.expand_dims(ezz[0], 0) for ezz in stats.Ezz])
        P2 = jnp.concat([jnp.expand_dims((ez[0] @ ez[0].mT), 0) for ez in stats.Ez])
        return jnp.atleast_2d(jnp.mean(P1 - P2, axis=0))

    def _transition_matrix_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        Numer = jnp.concat([jnp.sum(ezz1, axis=0, keepdims=True) for ezz1 in stats.Ezz1])
        Denom = jnp.concat([jnp.sum(ezz, axis=0, keepdims=True) for ezz in stats.Ezz])
        A = hseu.invmul(Numer, Denom)
        return jnp.atleast_2d(jnp.mean(A, axis=0))

    def _transition_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        P1 = [ezz[1:] for ezz in stats.Ezz]
        P2 = [ezz1 @ self.transition_matrices.T for ezz1 in stats.Ezz1]
        P3 = [p2.mT for p2 in P2]
        P4 = [self.transition_matrices @ ezz[:-1] @ self.transition_matrices.T for ezz in stats.Ezz]
        Gamma = jnp.concat([jnp.sum(p1-p2-p3+p4, axis=0, keepdims=True) / len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return jnp.atleast_2d(jnp.mean(Gamma, axis=0))

    def _observation_matrix_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        Numer = jnp.concat([jnp.sum(exz, axis=0, keepdims=True) for exz in stats.Exz])
        Denom = jnp.concat([jnp.sum(ezz, axis=0, keepdims=True) for ezz in stats.Ezz])
        C = hseu.invmul(Numer, Denom)
        return jnp.atleast_2d(jnp.mean(C, axis=0))

    def _observation_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> jax.Array:
        P1 = stats.Exx
        P2 = [self.observation_matrices @ ezx for ezx in stats.Ezx]
        P3 = [p2.mT for p2 in P2]
        P4 = [self.observation_matrices @ ezz @ self.observation_matrices.T for ezz in stats.Ezz]
        Sigma = jnp.concat([jnp.sum(p1-p2-p3+p4, axis=0, keepdims=True)/len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return jnp.atleast_2d(jnp.mean(Sigma, axis=0))

    def _em_mle(self, values: KalmanResults, stats: KalmanStatistics, normalize: bool) -> jax.Array:
        """Maximum likelihood estimation of all relevant parameters.

        Args:
            values (hsem.KalmanResults): Kalman filter results.
            stats (hsem.KalmanStatistics): Sufficient statistics from the Kalman filter/smoother.
            normalize (bool): If True, normalize the transition and observation matrices.

        Returns:
            jax.Array: The final negative log likelihood.
        """
        self.transition_matrices      = self._transition_matrix_mle(values, stats)
        if normalize:
            self.transition_matrices /= jnp.sum(self.transition_matrices, axis=1, keepdims=True)
        self.transition_covariance    = self._transition_covariance_mle(values, stats)
        self.observation_matrices     = self._observation_matrix_mle(values, stats)
        if normalize:
            self.observation_matrices /= jnp.sum(self.observation_matrices, axis=1, keepdims=True)
        self.observation_covariance   = self._observation_covariance_mle(values, stats)
        self.initial_state_mean       = self._initial_mean_mle(values, stats)
        self.initial_state_covariance = self._initial_covariance_mle(values, stats)
        return self._loglikelihood(values, stats)

    def _fit_ag(self, 
        loss_closure,
        loss_params,
        optimizer: str = "Adam",
        lr: float = .01,
        n_epochs: int = 1000, 
        gd_tol: float = 1e-3,
        **loss_args
    ):
        optimizers = {
            'Adam'  : optax.adam,
            'SGD'   : optax.sgd,
            'AdamW' : optax.adamw,
            'LBFGS' : optax.lbfgs
        }
        optimizer = optimizers[optimizer](learning_rate=lr)
        opt_state = optimizer.init(loss_params)
        prev_loss = np.inf 

        for epoch in range(n_epochs):
            loss,grads = jax.value_and_grad(loss_closure)(loss_params, **loss_args)
            updates,opt_state = optimizer.update(grads, opt_state, loss_params)
            params = optax.apply_updates(loss_params, updates)
            if epoch > 0 and abs((loss - prev_loss) / prev_loss) < gd_tol:
                break
            prev_loss = loss

        return params,jnp.array(prev_loss)

    def _em_autograd(self, 
            values: KalmanResults, 
            stats: KalmanStatistics, 
            normalize: bool, 
            n_epochs: int = 1000, 
            lr: float = .01, 
            gd_tol: float = 1e-3, 
        ) -> jax.Array:
        """Perform maximum likelihood estimation using autograd.

        Args:
            values (KalmanResults): Results of the Kalman filter/smoother.
            stats (KalmanStatistics): Sufficient statistics from the Kalman filter/smoother.
            normalize (bool): Whether or not to normalize the transition and observation matrices.
            n_epochs (int): Number of epochs for SGD.
            lr (float): Learning rate for the optimizer.
            gd_tol (float): Tolerance for SGD.

        Returns:
            jax.Array: The final negative log likelihood.
        """

        A = torch.zeros(self.augmented_dim, self.augmented_dim, requires_grad=True)
        C = torch.zeros(self.augmented_dim, self.obs_dim, requires_grad=True)
        ChGamma = torch.zeros(self.augmented_dim, self.augmented_dim, requires_grad=True)
        ChSigma = torch.zeros(self.augmented_dim, self.obs_dim, requires_grad=True)
        with torch.no_grad():
            A.copy_(self.transition_matrices)
            C.copy_(self.observation_matrices)
            ChGamma.copy_(torch.linalg.cholesky(self.transition_covariance))
            ChSigma.copy_(torch.linalg.cholesky(self.observation_covariance))

        optimizer = torch.optim.Adam(
            params=[A, C, ChGamma, ChSigma],
            lr=lr
        )
        prev_loss = 0

        # Just use MLEs for mean and covariance.
        # This way we have accurate estimates but still less to compute.
        self.initial_state_mean = self._initial_mean_mle(values, stats)
        self.initial_state_covariance = self._initial_covariance_mle(values, stats)

        n_batches = len(values.observations)
        
        for epoch in range(n_epochs):
            loss = 0 
            At = A 
            Ct = C 
            if normalize:
                At = At / At.sum(axis=1, keepdim=True)
                Ct = Ct / Ct.sum(axis=1, keepdim=True)
            Gamma = ChGamma @ ChGamma.mT
            Sigma = ChSigma @ ChSigma.mT
            
            for b in range(n_batches):
                iloss = 0

                tloss = stats.Ezz[b][1:] - stats.Ezz1[b] @ At.mT - At @ stats.Ez1z[b] + At @ stats.Ezz[b][:-1] @ At.mT
                tloss = torch.sum(tloss, axis=0) 
                tloss = torch.linalg.solve(Gamma, tloss)
                iloss += torch.trace(tloss)

                eloss = stats.Exx[b] - stats.Exz[b] @ Ct.mT - Ct @ stats.Ezx[b] + Ct @ stats.Ezz[b] @ Ct.mT
                eloss = torch.sum(eloss, axis=0)
                eloss = torch.linalg.solve(Sigma, eloss)
                iloss += torch.trace(eloss)

                iloss += hseu.logdet(Gamma) * (values.observations[b].shape[0] - 1)
                iloss += hseu.logdet(Sigma) * (values.observations[b].shape[0])
                iloss /= 2

                loss += iloss / n_batches

            if epoch > 0 and abs(loss.item() - prev_loss) < gd_tol:
                break
            prev_loss = loss.item() 

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        self.transition_matrices    = A.detach()
        if normalize:
            self.transition_matrices   /= self.transition_matrices.sum(axis=1, keepdims=True)
        self.observation_matrices   = C.detach()
        if normalize:
            self.observation_matrices  /= self.observation_matrices.sum(axis=1, keepdims=True)
        self.transition_covariance  = (ChGamma @ ChGamma.T).detach()
        self.observation_covariance = (ChSigma @ ChSigma.T).detach()

        return jnp.array(prev_loss)


    def _em(self, values: KalmanResults, normalize: bool, maximization_type: str = 'autograd', **autograd_args) -> jax.Array:
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            values (KalmanResults): Kalman filter results.
            normalize (bool): If True, normalize the transition and observation matrices.
            maximization_type (str): Type of maximization algorithm to use. Can be either 'mle' or 'autograd'.
            **autograd_args: Keyword arguments for the autograd maximization algorithm.

        Returns:
            jax.Array: The negative log likelihood of the data given the model parameters.
        """
        assert maximization_type in ['mle', 'autograd']
        stats = self._calc_sufficient_stats(values)
        if maximization_type == 'mle':
            return self._em_mle(values, stats, normalize)
        elif maximization_type == 'autograd':
            return self._em_autograd(values, stats, normalize, **autograd_args)

    def _calculate_marginals(self, environment_size, bin_size, values: KalmanResults) -> jax.Array:
        r"""Calculates the marginal probabilities for each bin in the environment.
        What is the probability that the mouse is in a given bin at a given time $P(X_t = x, Y_t = y|\mu_t, \Sigma_t)$

        Args:
            values (KalmanResults): Kalman filter results.

        Returns:
            jax.Array: The marginal probabilities for each bin in the environment. (Ncells, nbx, nby)
        """
        Lx = int((environment_size[2] - environment_size[0]) / bin_size)
        Ly = int((environment_size[3] - environment_size[1]) / bin_size)
        Z = hseu.create_grid(environment_size, bin_size)
        cumulative_probabilities = np.zeros((len(values.smoothed_mean), Lx, Ly))

        for i in range(len(values.smoothed_mean)):
            sm = values.smoothed_mean[i][:,:2]
            sc = values.smoothed_cov[i][:,:2,:2]
            _cp = np.zeros((sm.shape[0], Lx, Ly))
            for t in range(sm.shape[0]):
                log_prob = multivariate_normal.logpdf(Z, sm[t].ravel(), sc[t])
                log_prob = log_prob.reshape(Lx, Ly)
                _cp[t] = log_prob
            
            _cp -= logsumexp(_cp, axis=(1, 2), keepdims=True)
            _cp = jnp.exp(_cp)
            cumulative_probabilities[i] = jnp.sum(_cp, axis=0)

        return jnp.array(cumulative_probabilities / cumulative_probabilities.sum(axis=(1, 2), keepdims=True))

    def fit(self, 
            X: List[jax.Array],
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'autograd', 
            normalize: bool = False, 
            **diff_args
        ) -> KalmanResults:
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            X (jax.Array): The observations to fit the model to. Each individual observation time-series can have variable length.
            n_iter (int, optional): The number of EM iterations to run. Defaults to 100.
            emtol (float, optional): The tolerance for the change in log-likelihood between iterations. Defaults to 1e-3.
            maximization_type (str, optional): The type of maximization to use. Can be either 'mle' or 'autograd'. Defaults to 'autograd'.
            normalize (bool, optional): Whether to normalize the transition and observation matrices. Defaults to True.
            **diff_args: Keyword arguments to pass to the `_em` method.

        Returns:
            KalmanResults: The results of the EM algorithm. Estimated sequences can be accessed from this class itself.
        """
        X = self._parse_observations(X)

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        values = self._initialize(X)

        for i in range(n_iter):
            values = self.filter(values)
            values = self.smooth(values)
            ll = self._em(
                values,
                normalize,
                maximization_type,
                **diff_args
            )

            values.loglike.append(-ll)
            if not jnp.isfinite(values.loglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.loglike[-1] - values.loglike[-2]) / values.loglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

        if hasattr(self, 'bin_size') and hasattr(self, 'environment_size'):
            values.cumulative_probabilities = self._calculate_marginals(
                self.environment_size,
                self.bin_size, 
                values
            )

        return values

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    key = jax.random.key(42)

    def norm(mat):
        return mat / mat.sum(axis=1, keepdims=True)
    def cov(mat):
        return mat @ mat.T

    nd = 1

    t_trans_mat = norm(jax.random.uniform(key, shape=(nd,nd), dtype=jnp.float32))
    key,skey = jax.random.split(key)
    t_trans_noise = cov(jax.random.normal(skey, shape=(nd,nd), dtype=jnp.float32))
    key,skey = jax.random.split(key)
    t_emission_mat = norm(jax.random.uniform(skey, shape=(nd,nd), dtype=jnp.float32))
    key,skey = jax.random.split(key)
    t_emission_noise = cov(jax.random.normal(skey, shape=(nd,nd), dtype=jnp.float32))
    key,skey = jax.random.split(key)
    t_mu0 = jax.random.normal(key, shape=(nd,1), dtype=jnp.float32)
    key,skey = jax.random.split(key)
    t_sig0 = cov(jax.random.normal(skey, shape=(nd,nd), dtype=jnp.float32))

    z = [] 
    x = []
    for i in range(3):
        n_points = (i + 1) * 400
        _z = np.zeros((n_points, nd, 1))
        _x = np.zeros((n_points, nd, 1))

        key,skey = jax.random.split(key)
        V = jnp.linalg.cholesky(t_emission_noise) @ jax.random.normal(skey, shape=(n_points, nd, 1))
        key,skey = jax.random.split(key)
        W = jnp.linalg.cholesky(t_trans_noise) @ jax.random.normal(skey, shape=(n_points - 1, nd, 1))

        key,skey = jax.random.split(key)
        _z[0] = t_mu0 + jnp.linalg.cholesky(t_sig0) @ jax.random.normal(skey, shape=(nd,1))
        _x[0] = t_emission_mat @ _z[0] + V[0]

        for i in range(1, n_points):
            _z[i] = t_trans_mat @ _z[i-1] + W[i-1]
            _x[i] = t_emission_mat @ _z[i] + V[i]
        x.append(_x)
        z.append(_z)

    model = KalmanFilter(nd,nd)
    values = model.fit(x)

    for _x,_z,fm,fc,sm,sc in zip(x, z, values.filtered_mean, values.filtered_cov, values.smoothed_mean, values.smoothed_cov):

        if nd == 1:
            plt.figure(figsize=(20,10))
            plt.plot(_x.squeeze(), c='g', label='noisy measurements', linestyle='--')
            plt.plot(_z.squeeze(), c='b', label='true position', linestyle='--')
            t = np.arange(len(_x))
            plt.errorbar(t, fm.squeeze(), 
                            yerr=jnp.sqrt(fc.squeeze()),
                            c='r', label='kalman filter',
                            linestyle='--', capsize=4)
            plt.errorbar(t, sm.squeeze(),
                            yerr=jnp.sqrt(sc.squeeze()), 
                            c='y', label='kalman smooth', 
                            linestyle='--', capsize=4)
        else:
            plt.figure(figsize=(20,20))
            plt.plot(_x[:,0], _x[:,1], c='g', label='noisy measurements', linestyle='--')
            plt.plot(_z[:,0], _z[:,1], c='b', label='true position', linestyle='--')
            plt.plot(fm[:,0], fm[:,1], c='r', label='kalman filter', linestyle='--')
            plt.plot(sm[:,0], sm[:,1], c='y', label='kalman smooth', linestyle='--')

        plt.legend()
        plt.show()

    print(model.transition_matrices, t_trans_mat)
    print(model.transition_covariance, t_trans_noise)
    print(model.observation_matrices, t_emission_mat)
    print(model.observation_covariance, t_emission_noise)
    print(model.initial_state_mean, t_mu0)
    print(model.initial_state_covariance, t_sig0)