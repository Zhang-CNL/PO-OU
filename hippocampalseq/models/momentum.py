import numpy as np
import pynapple as nap
import torch
from dataclasses import dataclass, field
from functools import wraps

import hippocampalseq.utils as hseu

from .kalman_filter import *

__all__ = [
    'Momentum',
    'MomentumResults'
]

@dataclass
class MomentumResults(KalmanResults):
    emission_probabilities : list[torch.Tensor] = field(default_factory=list)
    approximate_mean       : list[torch.Tensor] = field(default_factory=list)
    approximate_covariance : list[torch.Tensor] = field(default_factory=list)

class Momentum(KalmanFilter):
    """State-space model that includes momentum as a parameters.
    Essentially, this collapses down to a second-order markov chain, so we can use kalman filtering.
    We have a uniform prior and our observation covariance shifts over time, so we take that into account here as well.
    """
    def __init__(
            self,
            place_fields: hseu.NDArray, 
            spikemat: list[hseu.NDArray],
            dt: float, 
            environment_size: list[tuple[int,...]],
            bin_size: int, 
            seed: int|None = 42
        ):
        r"""Initialize the momentum SSM.
        One-dimensional form of the model where we use the following approximation:

        $$\frac{d}{dt}\begin{pmatrix} v_t \\ z_t \end{pmatrix} = 
            \begin{pmatrix} -\lambda & 0 \\ 1 & 0\end{pmatrix}\begin{pmatrix} v_t \\ z_t \end{pmatrix}
            + \begin{pmatrix} \sigma_v & 0 \\ 0 & 0\end{pmatrix}\xi_t$$
            
            $v_t$ and $z_t$ are the velocity and position respectively, and each has an x and a y component.
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (list[tuple[int,...]]): List of coordinates corresponding to the bounds of the environment.
            bin_size (int): Size of individual bins in cm.
            seed: (int|None): Seed for the random number generator
        """
        if place_fields.shape[-1] == 1:
            n_zdim = 1
        else:
            n_zdim = 2
        super().__init__(n_zdim, n_zdim, 2)

        self.dt            = torch.tensor(dt)
        self.environment_size = environment_size
        self.bin_size = bin_size
        self.grid = hseu.make_ndgrid(self.environment_size, self.bin_size, indexing='ij')

        if seed is not None:
            torch.random.manual_seed(seed)

        if isinstance(place_fields, np.ndarray):
            place_fields = torch.from_numpy(place_fields)


        self.emission_probabilities = []
        self.approximate_mean       = []
        self.approximate_covariance = []
        for spk in spikemat:
            ep = torch.from_numpy(spk).double()
            #ep = ep[ep.sum(axis=1) > 0]
            emission_probability = hseu.calc_poisson_emission_probabilities_2d(
                ep, 
                place_fields,
                self.dt
            )
            emission_probability /= torch.sum(emission_probability, axis=(1,2), keepdim=True)
            emission_probability = torch.nan_to_num(emission_probability, nan=0.0, posinf=0.0, neginf=0.0)

            approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
                self.grid,
                emission_probability,
                'weighted',
            )
            
            self.emission_probabilities.append(emission_probability)
            self.approximate_mean.append(approx_mean)
            self.approximate_covariance.append(approx_cov)


        # Random initialization of parameters
        # Scale of ln(10) meters
        self.decay        = torch.rand(1)
        self.diffusion    = torch.rand(1)
        self.n_parameters = 2

    @wraps(KalmanFilter._initialize)
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
        r"""Construct the transition matrix in the form

        $$\begin{pmatrix}-\lambda\Delta t + 1& 0 \\ \Delta t &1\end{pmatrix}$$

        Args:
            decay (torch.Tensor): Decay parameter
        
        Returns:
            torch.Tensor: Transition matrix
        """
        I  = torch.eye(self.latent_dim)
        Z  = torch.zeros(self.latent_dim, self.latent_dim)
        If = torch.eye(self.augmented_dim)

        M1 = -decay * I
        top = torch.cat((M1, Z), dim=1)
        bottom = torch.cat((I , Z), dim=1)
        A = torch.cat((top, bottom), dim=0) * self.dt + If
        return A

    def _construct_transition_cov(self, diffusion: torch.Tensor) -> torch.Tensor:
        r"""Construct the transition covariance matrix in the form

        $$\begin{pmatrix}\sigma_v\sqrt{\Delta t} & 0 \\ 0 & 0\end{pmatrix}$$

        Args:
            diffusion (torch.Tensor): Diffusion parameter
        
        Returns:
            torch.Tensor: Transition covariance
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        sigma_m = diffusion**2 * self.dt * I
        top = torch.cat((sigma_m, Z), dim=1)
        bottom = torch.cat((Z, Z), dim=1)
        Gamma = torch.cat((top, bottom), dim=0)
        return Gamma

    def _init_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        $$P(z_1) = U(N_x,N_y) \approx \mathcal{N}\left(
            \begin{bmatrix}N_x/2 \\ N_y/2\end{bmatrix},
            \begin{bmatrix}N_x^2/12 & 0 \\ 0 & N_y^2/12
            \end{bmatrix}\right)$$

        The prior for the velocity is a stationary OU process prior:
            $$P(v_1) = \mathcal{N}(v_1|0, \sigma^2 / (2\lambda))$$
        Returns:
            (torch.Tensor): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): Prior covariance for augmented state
        """
        diffs = torch.tensor([es[1] + es[0] for es in self.environment_size])
        starts = torch.tensor([es[0] for es in self.environment_size])
        init_mean_z = (diffs / 2 + starts)[:,None]
        init_cov_z = torch.diag(diffs)**2 / 12

        sigma = torch.exp(self.diffusion)
        lmb   = torch.exp(self.decay)
        init_cov_v = torch.eye(self.latent_dim) * (sigma**2 * self.dt / (1 - (1 - lmb*self.dt)**2))
        init_mean_v = torch.zeros((self.latent_dim, 1))

        Z = torch.zeros(self.latent_dim, self.latent_dim)

        init_mean = torch.cat((init_mean_v, init_mean_z), dim=0)
        init_cov = torch.cat((
            torch.cat((init_cov_v, Z), dim=1),
            torch.cat((Z, init_cov_z), dim=1)
        ), dim=0)

        return init_mean, init_cov

    def _init_transition_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $(v_t \;  z_t)^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(torch.exp(self.decay))
        Q = self._construct_transition_cov(torch.exp(self.diffusion)) 
        b = torch.zeros(self.augmented_dim, 1)
        return A,Q,b

    def _init_observation_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Build the observation transition matrix and noise matrix.
        The noise is computed analytically from the data, while the transition
        matrix has the form $$\begin{bmatrix}0_2 & I_2\end{bmatrix}$$
        """
        I = torch.eye(self.obs_dim)
        Z = torch.zeros(self.obs_dim, self.latent_dim)
        H = torch.hstack((Z, I))
        R = self.approximate_covariance
        d = torch.zeros(self.obs_dim, 1)
        return H,R,d

    @wraps(KalmanFilter.filter)
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
        N = self.latent_dim
        btrace = torch.vmap(torch.trace)
        Cov,Ez,Ezz,Ezz1 = [],[],[],[]
        for sm,sc,sg in zip(values.smoothed_mean, values.smoothed_cov, values.smoothed_gain):
            muvv = sm[:,:N]
            v_vv = sc[:,:N,:N]
            g_vv = sg[:,:N,:N]
            # $$Cov[z_t,z_{t-1}] = \hat{V}_t J_{t-1}$$
            cov = btrace(v_vv[1:] @ g_vv[:-1].mT).squeeze() # tr(2,2) -> (1,1)
            # $$E[z_t] = \hat{\mu}_t$$
            ez = muvv # (2,1)
            # $$E[z_t^T z_t] = \hat{\mu}_t^T \hat{\mu}_t + tr(\hat{V}_t)$$
            ezz = (muvv.mT @ muvv).squeeze() + btrace(v_vv).squeeze() # (1,2)(2,1) + (1,1) -> (1,1)
            # $$E[z_t^T z_{t-1}] = tr(Cov[z_t,z_{t-1}]) + \hat{\mu}_t^T \hat{\mu}_{t-1}$$
            ezz1 = cov.squeeze() + (muvv[1:].mT @ muvv[:-1]).squeeze() # (1,1) + (1,2)(2,1) -> (1,1)

            Cov.append(cov)
            Ez.append(ez)
            Ezz.append(ezz)
            Ezz1.append(ezz1)

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

    @wraps(KalmanFilter._observed_loglikelihood)
    def _observed_loglikelihood(self, values: KalmanResults) -> torch.Tensor:
        log2pi = torch.log(2*PI)
        rank = self.augmented_dim
        loglike = 0

        for b in range(len(values.observations)):
            T = values.observations[b].shape[0]
            innovation = values.observations[b] - self.observation_matrices @ values.predicted_mean[b]
            innovation_cov = self.observation_matrices @ values.predicted_cov[b] @ self.observation_matrices.mT + self.observation_covariance[b]

            L = torch.linalg.cholesky(innovation_cov)
            alpha = torch.cholesky_solve(innovation, L)

            loglike += T * rank * log2pi \
                + 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1))) \
                + torch.sum(innovation.mT @ alpha, axis=0)

        return -0.5 * loglike.squeeze()

    def _loglikelihood(self, values: MomentumResults, stats: KalmanStatistics) -> torch.Tensor:
        r"""Calculate the full-data log-likelihood for the model given the current state.

        Args:
            values (MomentumResults): The current decoded values for hidden states.
            _ (KalmanStatistics): Ignored for this model.

        Returns:
            torch.Tensor: The log-likelihood for the model.
        """
        loglike = 0

        log2pi = torch.log(2*PI)
        rank   = self.latent_dim

        mu0 = self.initial_state_mean
        A   = self.transition_matrices
        C   = self.observation_matrices
        V0     = self.initial_state_covariance
        Sigma  = self.observation_covariance
        Gamma  = self.transition_covariance

        GammaS = torch.linalg.svd(Gamma)[1]
        GPi = torch.diag(1 / GammaS[:rank])
        GPld= torch.log(torch.prod(GammaS[:rank]))

        for b in range(len(values.observations)):
            T = values.observations[b].shape[0]

            muhat = values.smoothed_mean[b]
            vhat  = values.smoothed_cov[b]
            Jhat  = values.smoothed_gain[b]
            
            Exx  = values.observations[b] @ values.observations[b].mT
            Ezz  = vhat + muhat @ muhat.mT
            Ezx  = muhat @ values.observations[b].mT
            Ezz1 = Jhat[:-1] @ vhat[1:] + muhat[1:] @ muhat[:-1].mT 

            _loglike = 2 * T * rank * log2pi \
                + torch.logdet(V0) \
                + (T-1) * GPld \
                + torch.sum(torch.logdet(Sigma[b])) 


            ip1 = Ezz[0]
            ip2 = mu0 @ muhat[0].mT
            ip3 = mu0 @ mu0.mT
            ill = hseu.mulinv(V0, ip1 - ip2 - ip2.mT + ip3)
            ill = torch.trace(ill) 
            _loglike += ill

            tp1 = Ezz[1:,:rank,:rank]
            tp2 = (Ezz1 @ A.mT)[:,:rank,:rank]
            tp3 = (A @ Ezz[:-1] @ A.mT)[:,:rank,:rank]
            tll = torch.sum(tp1 - tp2 - tp2.mT + tp3, axis=0) 
            tll = GPi @ tll
            tll = torch.trace(tll) 
            _loglike += tll

            ep1 = Exx
            ep2 = C @ Ezx
            ep3 = C @ Ezz @ C.mT
            ell = hseu.mulinv(Sigma[b], ep1 - ep2 - ep2.mT + ep3)
            ell = torch.sum(ell, axis=0)
            ell = torch.trace(ell) 
            _loglike += ell

            loglike += _loglike
        return -loglike / 2.0

    def _solve_parameters(
            self, 
            values: MomentumResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
        ) -> torch.Tensor:
        """Perform maximum likelihood estimation of all relevant parameters for the momentum SSM using autograd.

        Args:
            values (MomentumResults): Momentum filtering pass results.
            stats (SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            optimizer (str): The optimizer to use.
            lr (float): Learning rate for the optimizer.
            n_epochs (int): Number of epochs for SGD.
            gd_tol (float): Tolerance for SGD.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        decay             = torch.zeros(1, requires_grad=True)
        diffusion         = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)

        params = [decay, diffusion]

        def loss_closure(params, n_batches: int, stats: SufficientStatistics):
            decay,diffusion = params

            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1
            
            lmb   = torch.exp(decay)
            sig   = torch.exp(diffusion)
            M     = 1 - lmb * self.dt 
            sigma = sig**2 * self.dt
            v0    = sigma / (1 - M**2)

            total_loss = 0
            for i in range(n_batches):
                T = len(Ezz[i])

                iloss = Ezz[i][0] / v0
                iloss = self.latent_dim * torch.log(v0) + iloss

                loss = Ezz[i][1:] - 2 * M * Ezz1[i] + M**2 * Ezz[i][:-1]
                loss = torch.sum(loss, axis=0) / sigma
                loss = self.latent_dim * (T-1) * torch.log(sigma) + loss

                total_loss += (iloss + loss) / 2 

            return total_loss

        loss,params = hseu.optimize(
            loss_closure, 
            params,
            {
                'n_batches' : len(values.observations), 
                'stats'     : stats,
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )
        
        self.decay     = params[0].detach()
        self.diffusion = params[1].detach()

        (
            self.transition_matrices,
            self.transition_covariance,
            self.transition_bias,
            self.observation_matrices,
            self.observation_covariance,
            self.observation_bias,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return loss

    def fit(self, 
            X=None, 
            n_iter: int = 1000, 
            emtol: float = 1e-3, 
            **maximization_args
        ) -> MomentumResults:
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            X (None): Value ignored. We treat self.approx_mean as the observed variable.
            n_iter (int): Number of EM iterations.
            emtol (float): Tolerance for the change in log-likelihood between iterations.
            **maximization_args: Keyword arguments to pass to the parent class's maximization method.

        Returns:
            MomentumResults: Results of fitting the model to the data.
        """
        return super().fit(
            self.approximate_mean,
            n_iter,
            emtol,
            **maximization_args
        )