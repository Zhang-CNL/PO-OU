import torch 
import numpy as np
import warnings
from torch.distributions import MultivariateNormal
from dataclasses import dataclass, field
from typing import Optional

import hippocampalseq.utils as hseu
from .statespace import *

__all__ = [
    'KalmanFilter',
    'KalmanResults',
    'KalmanStatistics',
    'PI'
]

PI = torch.tensor(np.pi)

@dataclass
class KalmanResults:
    observations   : list[torch.Tensor] = field(default_factory=list)
    predicted_mean : list[torch.Tensor] = field(default_factory=list)
    predicted_cov  : list[torch.Tensor] = field(default_factory=list)
    filtered_mean  : list[torch.Tensor] = field(default_factory=list)
    filtered_cov   : list[torch.Tensor] = field(default_factory=list)
    smoothed_gain  : list[torch.Tensor] = field(default_factory=list)
    smoothed_mean  : list[torch.Tensor] = field(default_factory=list)
    smoothed_cov   : list[torch.Tensor] = field(default_factory=list)
    loglike        : list[float]  = field(default_factory=list)
    loglike_full   : torch.Tensor = field(default_factory=lambda: torch.empty(0))
    aic            : float = 0
    bic            : float = 0
    cumulative_probabilities : torch.Tensor = field(default_factory=lambda: torch.empty(0))

@dataclass
class KalmanStatistics:
    Cov  : list[torch.Tensor] # $\hat{V}_tJ_{t-1}$
    Ez   : list[torch.Tensor] # $\mathbb{E}[z^T]$
    Ezz  : list[torch.Tensor] # $\mathbb{E}[zz^T]$
    Ezz1 : list[torch.Tensor] # $\mathbb{E}[z_{t}z_{t-1}^T]$
    Ez1z : list[torch.Tensor] # $\mathbb{E}[z_{t-1}z_t^T]$
    Exx  : list[torch.Tensor] # $\mathbb{E}[xx^T]  $
    Exz  : list[torch.Tensor] # $\mathbb{E}[xz^T]  $
    Ezx  : list[torch.Tensor] # $\mathbb{E}[zx^T]$

class KalmanFilter(StateSpace):
    """Base state space modeling class that implements a kalman filter"""
    def __init__(self, latent_dim: int, obs_dim: int, order: int = 1):
        """Initialize the kalman filter

        Args:
            latent_dim (int): Dimension of the latent state.
            obs_dim (int): Dimension of the observations.
            order (int, optional): Order of the state space model. The order of the markov chain being used. Defaults to 1.
        """
        torch.set_default_dtype(torch.double)

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

    def _parse_observations(self, obs: torch.Tensor|list[torch.Tensor]|None) -> list[torch.Tensor]|None:
        """Safely convert observations to their expected format."""
        if obs is None:
            return obs
        if not isinstance(obs, list):
            obs = [obs]
        for i in range(len(obs)):
            obs[i] = torch.atleast_3d(obs[i])
            if obs[i].shape[1] < obs[i].shape[2]:
                obs[i] = obs[i].mT
        return obs

    def _filter_init(self, values: KalmanResults, batch: int) -> KalmanResults:
        """Initialize the filter for the first observation."""
        P0Ct = self.initial_state_covariance @ self.observation_matrices.T 
        K1 = hseu.invmul(P0Ct, self.observation_matrices @ P0Ct + self.observation_covariance) 
        mu1 = self.initial_state_mean + K1 @ (values.observations[batch][0] - self.observation_matrices @ self.initial_state_mean)
        v1 = (torch.eye(self.augmented_dim) - K1 @ self.observation_matrices) @ self.initial_state_covariance
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
        vt  = (torch.eye(self.augmented_dim) - K @ self.observation_matrices) @ Pn1

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
                values = self._filter(values, batch, t)

        return values

    def _smooth_init(self, values: KalmanResults, batch: int) -> KalmanResults:
        """Initialize the RTS smoother."""
        values.smoothed_mean[batch][-1] = values.filtered_mean[batch][-1]
        values.smoothed_cov[batch][-1]  = values.filtered_cov[batch][-1]
        return values

    def _smooth(self, values: KalmanResults, batch: int, t: int) -> KalmanResults:
        """Run the RTS smoother for a single time step."""
        Amt   = values.predicted_mean[batch][t]
        Pt    = values.predicted_cov[batch][t]

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
                values = self._smooth(values, batch, t)
        return values

    def _init_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        m = torch.randn(self.augmented_dim, 1)
        n = torch.eye(self.augmented_dim)
        return m,n

    def _init_transition_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        F = torch.rand(self.augmented_dim, self.augmented_dim)
        F = F / F.sum(axis=1, keepdim=True)
        Q = torch.randn(self.augmented_dim, self.augmented_dim)
        Q = Q @ Q.T 
        return F,Q

    def _init_observation_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        H = torch.rand(self.augmented_dim, self.obs_dim)
        H = H / H.sum(axis=1, keepdim=True)
        R = torch.randn(self.obs_dim, self.obs_dim)
        R = R @ R.T
        return H, R

    def _initialize_parameters(self) -> tuple[torch.Tensor,...]:
        initial_mean, initial_cov = self._init_priors()
        trans_mat, trans_cov      = self._init_transition_matrices()
        obs_mat, obs_cov          = self._init_observation_matrices()
        return trans_mat, trans_cov, obs_mat, obs_cov, initial_mean, initial_cov

    def _initialize(self, X: list[torch.Tensor]) -> KalmanResults:
        tf_base  = [torch.zeros_like(x) for x in X]
        cov_base = [torch.zeros(x.shape[:-1] + (self.augmented_dim,)) for x in X]
        def _copy(x):
            return [i.clone() for i in x]
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

    def _loglikelihood(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        """Calculate the log likelihood of the model given the sufficient statistics and current 
        parameters.

        Args:
            values (KalmanResults): The filtered and smoothed values of the model.
            stats (KalmanStatistics): The sufficient statistics of the model.

        Returns:
            torch.Tensor: The log likelihood of the model.
        """
        ll = 0
        for b in range(len(values.observations)):
            T = values.observations[b].shape[0]

            _ll = 0
            _ll += torch.logdet(self.initial_state_covariance)
            _ll += torch.logdet(self.transition_covariance) * (T - 1)
            _ll += torch.logdet(self.observation_covariance) * T

            ip1 = stats.Ezz[b][0]
            ip2 = self.initial_state_mean @ values.smoothed_mean[b][0].mT
            ip3 = values.smoothed_mean[b][0] @ self.initial_state_mean.mT
            ip4 = self.initial_state_mean @ self.initial_state_mean.mT
            ill = hseu.mulinv(self.initial_state_covariance, ip1 - ip2 - ip3 + ip4)
            ill = torch.trace(ill) / 2
            _ll += ill

            tp1 = stats.Ezz[b][1:]
            tp2 = stats.Ezz1[b] @ self.transition_matrices.mT
            tp3 = tp2.mT 
            tp4 = self.transition_matrices @ stats.Ezz[b][:-1] @ self.transition_matrices.mT
            tll = torch.sum(tp1 - tp2 - tp3 + tp4, axis=0) 
            tll = hseu.mulinv(self.transition_covariance, tll)
            tll = torch.trace(tll) / 2
            _ll += tll

            ep1 = stats.Exx[b]
            ep2 = self.observation_matrices @ stats.Ezx[b]
            ep3 = ep2.mT
            ep4 = self.observation_matrices @ stats.Ezz[b] @ self.observation_matrices.mT
            ell = torch.sum(ep1 - ep2 - ep3 + ep4, axis=0)
            ell = hseu.mulinv(self.observation_covariance, ell)
            ell = torch.trace(ell) / 2
            _ll += ell

            _ll += T * self.latent_dim * torch.log(2*PI)
            ll += _ll / 2
        return ll

    def _initial_mean_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        return torch.atleast_2d(torch.mean(torch.cat([sm[0].unsqueeze(0) for sm in values.smoothed_mean]), axis=0))
    
    def _initial_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        """
        mu1 = torch.cat([sm[0].unsqueeze(0) for sm in values.smoothed_mean])
        c1  = torch.cat([sc[0].unsqueeze(0) for sc in values.smoothed_cov])
        P1 = c1 + mu1 @ mu1.mT 
        P2 = self.initial_state_mean @ mu1.mT 
        P3 = P2.mT 
        P4 = self.initial_state_mean @ self.initial_state_mean.mT 
        return torch.atleast_2d(torch.mean(P1 - P2 - P3 + P4, axis=0))
        """ 
        P1 = torch.cat([ezz[0].unsqueeze(0) for ezz in stats.Ezz])
        P2 = torch.cat([(ez[0] @ ez[0].mT).unsqueeze(0) for ez in stats.Ez])
        return torch.atleast_2d(torch.mean(P1 - P2, axis=0))

    def _transition_matrix_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        Numer = torch.cat([torch.sum(ezz1, axis=0, keepdim=True) for ezz1 in stats.Ezz1])
        Denom = torch.cat([torch.sum(ezz, axis=0, keepdim=True) for ezz in stats.Ezz])
        A = hseu.invmul(Numer, Denom)
        return torch.atleast_2d(torch.mean(A, axis=0))

    def _transition_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        P1 = [ezz[1:] for ezz in stats.Ezz]
        P2 = [ezz1 @ self.transition_matrices.T for ezz1 in stats.Ezz1]
        P3 = [p2.mT for p2 in P2]
        P4 = [self.transition_matrices @ ezz[:-1] @ self.transition_matrices.T for ezz in stats.Ezz]
        Gamma = torch.cat([torch.sum(p1-p2-p3+p4, axis=0, keepdim=True) / len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return torch.atleast_2d(torch.mean(Gamma, axis=0))

    def _observation_matrix_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        Numer = torch.cat([torch.sum(exz, axis=0, keepdim=True) for exz in stats.Exz])
        Denom = torch.cat([torch.sum(ezz, axis=0, keepdim=True) for ezz in stats.Ezz])
        C = hseu.invmul(Numer, Denom)
        return torch.atleast_2d(torch.mean(C, axis=0))

    def _observation_covariance_mle(self, values: KalmanResults, stats: KalmanStatistics) -> torch.Tensor:
        P1 = stats.Exx
        P2 = [self.observation_matrices @ ezx for ezx in stats.Ezx]
        P3 = [p2.mT for p2 in P2]
        P4 = [self.observation_matrices @ ezz @ self.observation_matrices.T for ezz in stats.Ezz]
        Sigma = torch.cat([torch.sum(p1-p2-p3+p4, axis=0, keepdim=True)/len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return torch.atleast_2d(torch.mean(Sigma, axis=0))

    def _solve_parameters(self, values: KalmanResults, stats: KalmanStatistics, **_) -> torch.Tensor:
        with torch.no_grad():
            self.transition_matrices      = self._transition_matrix_mle(values, stats)
            self.transition_covariance    = self._transition_covariance_mle(values, stats)
            self.observation_matrices     = self._observation_matrix_mle(values, stats)
            self.observation_covariance   = self._observation_covariance_mle(values, stats)
            self.initial_state_mean       = self._initial_mean_mle(values, stats)
            self.initial_state_covariance = self._initial_covariance_mle(values, stats)

            return self._loglikelihood(values, stats)


    def _maximize(self, values: KalmanResults, **kwargs) -> torch.Tensor:
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            values (KalmanResults): Kalman filter results.
            **kwargs: Keyword arguments for solving for parameters.

        Returns:
            torch.Tensor: The negative log likelihood of the data given the model parameters.
        """
        with torch.no_grad():
            stats = self._calc_sufficient_stats(values)
        return self._solve_parameters(values, stats, **kwargs)

    def _calculate_marginals(self, environment_size, bin_size, values: KalmanResults) -> torch.Tensor:
        r"""Calculates the marginal probabilities for each bin in the environment.
        What is the probability that the mouse is in a given bin at a given time $P(X_t = x, Y_t = y|\mu_t, \Sigma_t)$

        Args:
            environment_size (tuple): Size of the environment. (xmin, ymin, xmax, ymax)
            bin_size (int): Size of individual bins in cm.
            values (KalmanResults): Kalman filter results.

        Returns:
            torch.Tensor: The marginal probabilities for each bin in the environment. (Ncells, nbx, nby)
        """
        sz = tuple(int((es[1] - es[0]) / bin_size) for es in environment_size)
        if len(sz) == 1:
            sz = sz + (1,)
        Z = hseu.make_ndgrid(environment_size, bin_size, indexing='ij')
        cumulative_probabilities = torch.zeros((len(values.smoothed_mean),) + sz)

        for i in range(len(values.smoothed_mean)):
            sm = values.smoothed_mean[i][:,:self.latent_dim]
            sc = torch.atleast_2d(values.smoothed_cov[i][:,:self.latent_dim,:self.latent_dim])
            cp = torch.zeros((sm.shape[0],)+sz)
            for t in range(sm.shape[0]):
                L = torch.linalg.cholesky(sc[t])
                mvn = MultivariateNormal(
                    sm[t].ravel(), 
                    scale_tril=L
                )
                log_prob = mvn.log_prob(Z)
                log_prob = log_prob.reshape(sz)
                cp[t] = log_prob
            
            cp -= torch.logsumexp(cp, dim=(1, 2), keepdim=True)
            cp = torch.exp(cp)
            cumulative_probabilities[i] = torch.sum(cp,axis=0)

        return cumulative_probabilities / cumulative_probabilities.sum(axis=(1, 2), keepdim=True)

    def fit(self, 
            X: list[torch.Tensor],
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            checkpoint_path: Optional[str] = None,
            **maximization_args
        ) -> KalmanResults:
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            X (torch.Tensor): The observations to fit the model to. Each individual observation time-series can have variable length.
            n_iter (int, optional): The number of EM iterations to run. Defaults to 100.
            emtol (float, optional): The tolerance for the change in log-likelihood between iterations. Defaults to 1e-3.
            checkpoint_path (Optional[str], optional): The path to save checkpoint files. Checkpoint files are deleted after a successful run. Defaults to None.
            **maximization_args: Keyword arguments to pass to the `_em` method.

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

        if checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)

        for i in range(n_iter):
            with torch.no_grad():
                values = self.filter(values)
                values = self.smooth(values)
            ll = self._maximize(
                values,
                **maximization_args
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
                    hseu.save_pickle(values, f"./{checkpoint_path}/model_values_epoch_{i}.pkl")
                    hseu.save_pickle(self, f"./{checkpoint_path}/_model_epoch_{i}.pkl")

        
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        stats = self._calc_sufficient_stats(values)
        values.loglike_full = self._loglikelihood(values, stats)
        values.aic = self.aic(values.loglike_full)
        values.bic = self.bic(
            values.loglike_full, 
            sum(len(obs) for obs in values.observations)
        )
        if hasattr(self, 'bin_size') and hasattr(self, 'environment_size'):
            values.cumulative_probabilities = self._calculate_marginals(
                self.environment_size,
                self.bin_size,
                values
            )

        return values


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.double)
    torch.manual_seed(42)

    def norm(mat):
        return mat / mat.sum(axis=1, keepdims=True)
    def cov(mat):
        return mat @ mat.T

    nd = 1

    t_trans_mat = norm(torch.rand(nd,nd))
    t_trans_noise = cov(torch.randn(nd,nd))
    t_emission_mat = norm(torch.rand(nd,nd))
    t_emission_noise = cov(torch.randn(nd,nd))
    t_mu0 = torch.randn(nd,1)
    t_sig0 = cov(torch.randn(nd,nd))

    z = [] 
    x = []
    for i in range(3):
        n_points = (i + 1) * 400
        _z = torch.zeros((n_points, nd, 1))
        _x = torch.zeros((n_points, nd, 1))

        V = torch.linalg.cholesky(t_emission_noise) @ torch.randn(n_points, nd, 1)
        W = torch.linalg.cholesky(t_trans_noise) @ torch.randn(n_points - 1, nd, 1)

        _z[0] = t_mu0 + torch.linalg.cholesky(t_sig0) @ torch.randn(nd,1)
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
                            yerr=torch.sqrt(fc.squeeze()),
                            c='r', label='kalman filter',
                            linestyle='--', capsize=4)
            plt.errorbar(t, sm.squeeze(),
                            yerr=torch.sqrt(sc.squeeze()), 
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