import torch 
import numpy as np
from dataclasses import dataclass

import hippocampalseq.utils as hseu

__all__ = [
    'StateSpaceModel',
    'KalmanResults',
    'SufficientStatistics',
    'PI'
]

PI = torch.tensor(np.pi)

@dataclass
class KalmanResults:
    observations  : torch.Tensor
    predicted_mean: torch.Tensor
    predicted_cov : torch.Tensor
    filtered_mean : torch.Tensor
    filtered_cov  : torch.Tensor
    smoothed_gain : torch.Tensor
    smoothed_mean : torch.Tensor
    smoothed_cov  : torch.Tensor
    negloglike    : list

@dataclass
class SufficientStatistics:
    Cov: torch.Tensor # $\hat{V}_tJ_{t-1}$
    Ez: torch.Tensor # $\mathbb{E}[z^T]$
    Ezz: torch.Tensor # $\mathbb{E}[zz^T]$
    Ezz1: torch.Tensor # $\mathbb{E}[z_{t}z_{t-1}^T]$
    Ez1z: torch.Tensor # $\mathbb{E}[z_{t-1}z_t^T]$
    Exx: torch.Tensor # $\mathbb{E}[xx^T]  $
    Exz: torch.Tensor # $\mathbb{E}[xz^T]  $
    Ezx: torch.Tensor # $\mathbb{E}[zx^T]$

class StateSpaceModel:
    """Base state space modeling class that implements a kalman filter"""
    def __init__(self, latent_dim, obs_dim, order=1):
        torch.set_default_dtype(torch.double)

        self.latent_dim    = latent_dim
        self.augmented_dim = order * latent_dim
        self.obs_dim       = obs_dim

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format."""
        obs = torch.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs

    def _filter_init(self, values: KalmanResults):
        """Initialize the filter for the first observation."""
        P0Ct = self.initial_state_covariance @ self.observation_matrices.T 
        K1 = hseu.invmul(P0Ct, self.observation_covariance @ P0Ct + self.observation_matrices) 
        mu1 = self.initial_state_mean + K1 @ (values.observations[0] - self.observation_matrices @ self.initial_state_mean)
        v1 = (torch.eye(self.augmented_dim) - K1 @ self.observation_matrices) @ self.initial_state_covariance

        values.filtered_mean[0]  = mu1
        values.filtered_cov[0]   = v1
        values.predicted_mean[0] = self.transition_matrices @ mu1
        values.predicted_cov[0]  = self.transition_matrices @ v1 @ self.transition_matrices.T + self.transition_covariance
        return values


    def _filter(self, values: KalmanResults, t: int):
        """Run the filter for a single time step."""
        Am1 = values.predicted_mean[t-1]
        Pn1 = values.predicted_cov[t-1]

        Pct = Pn1 @ self.observation_matrices.T
        K = hseu.invmul(Pct, self.observation_matrices @ Pct + self.observation_covariance)

        mut = Am1 + K @ (values.observations[t] - self.observation_matrices @ Am1)
        vt  = (torch.eye(self.augmented_dim) - K @ self.observation_matrices) @ Pn1

        Am = self.transition_matrices @ mut
        Pn = self.transition_matrices @ vt @ self.transition_matrices.T + self.transition_covariance

        values.filtered_mean[t]  = mut
        values.filtered_cov[t]   = vt
        values.predicted_mean[t] = Am
        values.predicted_cov[t]  = Pn
        return values

    def filter(self, values: KalmanResults):
        """Run the filter."""
        # Initialize the filter
        values = self._filter_init(values)

        for t in range(values.observations.shape[0]):
            values = self._filter(values, t)
        return values

    def _smooth_init(self, values: KalmanResults):
        """Initialize the RTS smoother."""
        values.smoothed_mean[-1] = values.filtered_mean[-1]
        values.smoothed_cov[-1] = values.filtered_cov[-1]
        return values

    def _smooth(self, values: KalmanResults, t: int):
        """Run the RTS smoother for a single time step."""
        Amt   = values.predicted_mean[t]
        Pt    = values.predicted_cov[t]

        J = hseu.invmul(values.filtered_cov[t] @ self.transition_matrices.T, Pt)
        muht = values.filtered_mean[t] + J @ (values.smoothed_mean[t+1] - Amt) 
        vht = values.filtered_cov[t] + J @ (values.smoothed_cov[t+1] - Pt) @ J.mT

        values.smoothed_gain[t] = J
        values.smoothed_mean[t] = muht
        values.smoothed_cov[t]  = vht
        return values

    def smooth(self, values: KalmanResults):
        """Run the RTS smoother."""
        # Initialize the RTS smoother
        values = self._smooth_init(values)
        for t in reversed(range(values.observations.shape[0] - 1)):
            values = self._smooth(values, t)
        return values

    def _init_priors(self):
        return torch.randn(self.augmented_dim, 1), torch.eye(self.augmented_dim)

    def _init_transition_matrices(self):
        F = torch.randn(self.augmented_dim, self.augmented_dim)
        F = F / F.sum(axis=1, keepdim=True)
        Q = torch.randn(self.augmented_dim, self.augmented_dim)
        Q = Q @ Q.T 
        return F,Q

    def _init_observation_matrices(self):
        H = torch.randn(self.augmented_dim, self.augmented_dim)
        H = H / H.sum(axis=1, keepdim=True)
        R = torch.randn(self.augmented_dim, self.obs_dim)
        R = R @ R.T
        return H, R

    def _initialize_parameters(self):
        initial_mean, initial_cov = self._init_priors()
        trans_mat, trans_cov     = self._init_transition_matrices()
        obs_mat, obs_cov         = self._init_observation_matrices()
        return trans_mat, trans_cov, obs_mat, obs_cov, initial_mean, initial_cov

    def _calc_sufficient_stats(self, values: KalmanResults):
        """Calculate sufficient statistics for performing maximization given the filtered
         and smoothed values of the model.

        Args:
            values (KalmanResults): The filtered and smoothed values of the model.

        Returns:
            SufficientStatistics: The sufficient statistics of the model.
        """
        Cov = values.smoothed_cov[1:] @ values.smoothed_gain[:-1].mT
        Ez = values.smoothed_mean
        Ezz = values.smoothed_cov + values.smoothed_mean @ values.smoothed_mean.mT
        Ezz1 = values.smoothed_mean[1:] @ values.smoothed_mean[:-1].mT + Cov
        Ez1z = Ezz1.mT
        Exx = values.observations @ values.observations.mT
        Exz = values.observations @ values.smoothed_mean.mT
        Ezx = Exz.mT
        return SufficientStatistics(
            Cov, Ez, Ezz, Ezz1, Ez1z, Exx, Exz, Ezx
        )

    def _loglikelihood(self, values: KalmanResults, stats: SufficientStatistics) -> torch.T:
        """Calculate the log likelihood of the model given the sufficient statistics and current 
        parameters.

        Args:
            values (KalmanResults): The filtered and smoothed values of the model.
            stats (SufficientStatistics): The sufficient statistics of the model.

        Returns:
            torch.Tensor: The log likelihood of the model.
        """
        T = values.observations.shape[0]
        ll = 0 
        ll += torch.logdet(self.initial_state_covariance)
        ll += torch.logdet(self.transition_covariance) * (T - 1)
        ll += torch.logdet(self.observation_covariance) * T
        ll /= 2 

        ip1 = stats.Ezz[0]
        ip2 = self.initial_state_mean @ values.smoothed_mean[0].mT
        ip3 = values.smoothed_mean[0] @ self.initial_state_mean.mT
        ip4 = self.initial_state_mean @ self.initial_state_mean.mT
        ill = hseu.mulinv(self.initial_state_covariance + .001 * torch.eye(self.augmented_dim), ip1 - ip2 - ip3 + ip4)
        ill = torch.trace(ill) / 2
        ll += ill

        tp1 = stats.Ezz[1:]
        tp2 = stats.Ezz1 @ self.transition_matrices.mT
        tp3 = tp2.mT 
        tp4 = self.transition_matrices @ stats.Ezz[:-1] @ self.transition_matrices.mT
        tll = torch.sum(tp1 - tp2 - tp3 + tp4, axis=0) 
        tll = hseu.mulinv(self.transition_covariance, tll)
        tll = torch.trace(tll) / 2
        ll += tll

        ep1 = stats.Exx 
        ep2 = self.observation_matrices @ stats.Ezx
        ep3 = ep2.mT
        ep4 = self.observation_matrices @ stats.Ezz @ self.observation_matrices.mT
        ell = torch.sum(ep1 - ep2 - ep3 + ep4, axis=0)
        ell = hseu.mulinv(self.observation_covariance, ell)
        ell = torch.trace(ell) / 2
        ll += ell

        ll += T * self.latent_dim * torch.log(PI)
        return ll

    def _initial_mean_mle(self, values: KalmanResults, stats: SufficientStatistics):
        return torch.atleast_2d(values.smoothed_mean[0])
    
    def _initial_covariance_mle(self, values: KalmanResults, stats: SufficientStatistics):
        mu1 = values.smoothed_mean[0]
        P1 = values.smoothed_cov[0] + mu1 @ mu1.mT
        P2 = self.initial_state_mean @ mu1.mT 
        P3 = P2.mT 
        P4 = self.initial_state_mean @ self.initial_state_mean.mT 
        return torch.atleast_2d(P1 - P2 - P3 + P4)

    def _transition_matrix_mle(self, values: KalmanResults, stats: SufficientStatistics):
        Numer = torch.sum(stats.Ezz1, axis=0)
        Denom = torch.sum(stats.Ezz, axis=0)
        return torch.atleast_2d(hseu.invmul(Numer, Denom))

    def _transition_covariance_mle(self, values: KalmanResults, stats: SufficientStatistics):
        P1 = stats.Ezz[1:]
        P2 = stats.Ezz1 @ self.transition_matrices.T
        P3 = P2.mT 
        P4 = self.transition_matrices @ stats.Ezz[:-1] @ self.transition_matrices.T
        emission = torch.sum(P1 - P2 - P3 + P4, axis=0) / (values.observations.shape[0] - 1)
        return torch.atleast_2d(emission) 

    def _observation_matrix_mle(self, values: KalmanResults, stats: SufficientStatistics):
        Numer = torch.sum(stats.Exz, axis=0)
        Denom = torch.sum(stats.Ezz, axis=0)
        return torch.atleast_2d(hseu.invmul(Numer, Denom))

    def _observation_covariance_mle(self, values: KalmanResults, stats: SufficientStatistics):
        P1 = stats.Exx
        P2 = self.observation_matrices @ stats.Ezx 
        P3 = P2.mT
        P4 = self.observation_matrices @ stats.Ezz @ self.observation_matrices.T
        observation = torch.sum(P1 - P2 - P3 + P4, axis=0) / (values.observations.shape[0])
        return torch.atleast_2d(observation)

    def _em_mle(self, values: KalmanResults, stats: SufficientStatistics, normalize: bool):
        """Maximum likelihood estimation of all relevant parameters.

        Args:
            values (hsem.KalmanResults): Kalman filter results.
            stats (hsem.SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            normalize (bool): If True, normalize the transition and observation matrices.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        with torch.no_grad():
            self.transition_matrices      = self._transition_matrix_mle(values, stats)
            if normalize:
                self.transition_matrices /= torch.sum(self.transition_matrices, axis=1, keepdim=True)
            self.transition_covariance    = self._transition_covariance_mle(values, stats)
            self.observation_matrices     = self._observation_matrix_mle(values, stats)
            if normalize:
                self.observation_matrices /= torch.sum(self.observation_matrices, axis=1, keepdim=True)
            self.observation_covariance   = self._observation_covariance_mle(values, stats)
            self.initial_state_mean       = self._initial_mean_mle(values, stats)
            self.initial_state_covariance = self._initial_covariance_mle(values, stats)
        return self._loglikelihood(values, stats)

    def _em_autograd(self, 
            values: KalmanResults, 
            stats: SufficientStatistics, 
            normalize: bool, 
            n_epochs: int = 1000, 
            lr: float = .01, 
            gd_tol: float = 1e-3, 
            seed: int = 42
        ) -> torch.Tensor:
        """Perform maximum likelihood estimation using autograd.

        Args:
            values (KalmanResults): Results of the Kalman filter/smoother.
            stats (SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            normalize (bool): Whether or not to normalize the transition and observation matrices.
            n_epochs (int): Number of epochs for SGD.
            lr (float): Learning rate for the optimizer.
            gd_tol (float): Tolerance for SGD.
            seed (int): Seed for the random number generator.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        if seed is not None: 
            torch.random.manual_seed(seed)

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

        self.initial_state_mean = self._initial_mean_mle(values, stats)
        self.initial_state_covariance = self._initial_covariance_mle(values, stats)
        
        for epoch in range(n_epochs):
            loss = 0 
            At = A 
            Ct = C 
            if normalize:
                At = At / At.sum(axis=1, keepdim=True)
                Ct = Ct / Ct.sum(axis=1, keepdim=True)
            Gamma = ChGamma @ ChGamma.mT
            Sigma = ChSigma @ ChSigma.mT

            tloss = stats.Ezz[1:] - stats.Ezz1 @ At.mT - At @ stats.Ez1z + At @ stats.Ezz[:-1] @ At.mT
            tloss = torch.sum(tloss, axis=0) 
            tloss = torch.linalg.solve(Gamma, tloss)
            loss += torch.trace(tloss)

            eloss = stats.Exx - stats.Exz @ Ct.mT - Ct @ stats.Ezx + Ct @ stats.Ezz @ Ct.mT
            eloss = torch.sum(eloss, axis=0)
            eloss = torch.linalg.solve(Sigma, eloss)
            loss += torch.trace(eloss)

            loss += torch.logdet(Gamma) * (values.observations.shape[0] - 1)
            loss += torch.logdet(Sigma) * (values.observations.shape[0])
            loss /= 2

            if epoch > 0 and abs(loss.item() - prev_loss) < gd_tol:
                break
            prev_loss = loss.item() 

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        self.transition_matrices    = A.detach()
        if normalize:
            self.transition_matrices   /= self.transition_matrices.sum(axis=1, keepdim=True)
        self.observation_matrices   = C.detach()
        if normalize:
            self.observation_matrices  /= self.observation_matrices.sum(axis=1, keepdim=True)
        self.transition_covariance  = (ChGamma @ ChGamma.T).detach()
        self.observation_covariance = (ChSigma @ ChSigma.T).detach()

        return torch.tensor(prev_loss)


    def _em(self, values: KalmanResults, normalize: bool, maximization_type: str = 'autograd', **autograd_args):
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            values (hsem.KalmanResults): Kalman filter results.
            normalize (bool): If True, normalize the transition and observation matrices.
            maximization_type (str): Type of maximization algorithm to use. Can be either 'mle' or 'autograd'.
            **autograd_args: Keyword arguments for the autograd maximization algorithm.

        Returns:
            torch.Tensor: The negative log likelihood of the data given the model parameters.
        """
        assert maximization_type in ['mle', 'autograd']
        stats = self._calc_sufficient_stats(values)
        if maximization_type == 'mle':
            return self._em_mle(values, stats, normalize)
        elif maximization_type == 'autograd':
            return self._em_autograd(values, stats, normalize, **autograd_args)

    def em(self, X: torch.Tensor, normalize: bool = True, n_iter: int = 100, emtol: float = 1e-3, **diff_args):
        """Expectation-Maximization (EM) algorithm for the state-space model.

        Args:
            X (torch.Tensor): The observations to fit the model to.
            normalize (bool, optional): Whether to normalize the transition and observation matrices. Defaults to True.
            n_iter (int, optional): The number of EM iterations to run. Defaults to 100.
            emtol (float, optional): The tolerance for the change in log-likelihood between iterations. Defaults to 1e-3.
            **diff_args: Keyword arguments to pass to the `_em` method.

        Returns:
            KalmanResults: The results of the EM algorithm. Estimated parameters can be accessed from this class itself.
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

        values = KalmanResults(
            observations  =X,
            predicted_mean=torch.zeros(X.shape[0], self.augmented_dim, 1),
            predicted_cov =torch.zeros(X.shape[0], self.augmented_dim, self.augmented_dim),
            filtered_mean =torch.zeros(X.shape[0], self.augmented_dim, 1),
            filtered_cov  =torch.zeros(X.shape[0], self.augmented_dim, self.augmented_dim),
            smoothed_gain =torch.zeros(X.shape[0], self.augmented_dim, self.augmented_dim),
            smoothed_mean =torch.zeros(X.shape[0], self.augmented_dim, 1),
            smoothed_cov  =torch.zeros(X.shape[0], self.augmented_dim, self.augmented_dim),
            negloglike    =[]
        )


        for i in range(n_iter):
            with torch.no_grad():
                values = self.filter(values)
                values = self.smooth(values)
            ll = self._em(
                values,
                normalize,
                **diff_args
            )

            values.negloglike.append(-ll)
            if torch.isnan(values.negloglike[-1]):
                print("Log-likelihood is NaN, stopping EM")
                break

            if i > 0 and abs((values.negloglike[-1] - values.negloglike[-2]) / values.negloglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

        return values


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    torch.set_default_dtype(torch.double)
    t_trans_mat = torch.tensor([[1.0]])
    t_trans_noise = torch.tensor([[0.5]])
    t_emission_mat = torch.tensor([[1.0]])
    t_emission_noise = torch.tensor([[0.5]])
    t_mu0 = torch.tensor([[0.0]])
    t_sig0 = torch.tensor([[1.0]])

    n_points = 1000
    torch.manual_seed(42)
    z = torch.zeros((n_points, 1, 1))
    x = torch.zeros((n_points, 1, 1))

    V = torch.linalg.cholesky(t_emission_noise) @ torch.randn(n_points, 1, 1)
    W = torch.linalg.cholesky(t_trans_noise) @ torch.randn(n_points - 1, 1, 1)

    z[0] = t_mu0 + torch.linalg.cholesky(t_sig0) @ torch.randn(1, 1)
    x[0] = t_emission_mat @ z[0] + V[0]

    for i in range(1, n_points):
        z[i] = t_trans_mat @ z[i-1] + W[i-1]
        x[i] = t_emission_mat @ z[i] + V[i]

    model = StateSpaceModel(1, 1)

    values = model.em(x)

    plt.figure(figsize=(20,10))
    plt.plot(x.squeeze(), c='g', label='noisy measurements', linestyle='--')
    plt.plot(z.squeeze(), c='b', label='true position', linestyle='--')
    t = np.arange(n_points)
    plt.errorbar(t, values.filtered_mean.squeeze(), 
                    yerr=torch.sqrt(values.filtered_cov.squeeze()),
                    c='r', label='kalman filter',
                    linestyle='--', capsize=4)
    plt.errorbar(t, values.smoothed_mean.squeeze(),
                    yerr=torch.sqrt(values.smoothed_cov.squeeze()), 
                    c='y', label='kalman smooth', 
                    linestyle='--', capsize=4)
    plt.legend()
    plt.show()
    print(model.transition_matrices)
    print(model.transition_covariance)
    print(model.observation_matrices)
    print(model.observation_covariance)
    print(model.initial_state_mean)
    print(model.initial_state_covariance)

