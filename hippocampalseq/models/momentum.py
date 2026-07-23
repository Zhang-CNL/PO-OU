import numpy as np
import pynapple as nap
import torch
from dataclasses import dataclass, field

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
            ep = ep[ep.sum(axis=1) > 0]
            emission_probability = hseu.calc_poisson_emission_probabilities_2d(
                ep, 
                place_fields,
                self.dt
            )
            emission_probability /= torch.sum(emission_probability, axis=(1,2), keepdim=True)

            approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
                self.grid,
                emission_probability,
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
        r"""Construct the transition matrix in the form

        $$\begin{pmatrix}-\lambda\Delta t + 1& 0 \\ \Delta t &1\end{pmatrix}$$

        Args:
            decay (torch.Tensor): Decay parameter
        
        Returns:
            torch.Tensor: Transition matrix
        """
        I  = torch.eye(self.obs_dim)
        Z  = torch.zeros(self.obs_dim, self.obs_dim)
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
        I = torch.eye(self.obs_dim)
        Z = torch.zeros((self.obs_dim, self.obs_dim))

        sigma_m = diffusion * torch.sqrt(self.dt) * I
        top = torch.cat((sigma_m, Z), dim=1)
        bottom = torch.cat((Z, Z), dim=1)
        Gamma = torch.cat((top, bottom), dim=0)
        return Gamma

    def _init_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        $$U(N_x,N_y) \approx \mathcal{N}\left(
            \begin{bmatrix}N_x/2 \\ N_y/2\end{bmatrix},
            \begin{bmatrix}N_x^2/12 & 0 \\ 0 & N_y^2/12
            \end{bmatrix}\right)$$
        Returns:
            (torch.Tensor): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): Prior covariance for augmented state
        """
        # $z_2 = I z_1 + \sigma_0^2dt\xi_1$
        diffs = torch.tensor([es[1] - es[0] for es in self.environment_size])
        starts = torch.tensor([es[0] for es in self.environment_size])
        init_mean = (diffs / 2 + starts)[:,None]
        init_cov = torch.diag(diffs)**2 / 12

        return init_mean, init_cov

    def _init_transition_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Construct transition matrices for momentum SSM.

        Returns:
            (torch.Tensor): transition matrix for augmented state $(v_t \;  z_t)^T$
            (torch.Tensor): process noise covariance for augmented state
        """
        A = self._construct_transition_mat(self.decay.exp())
        Q = self._construct_transition_cov(self.diffusion.exp()) 

        return A,Q

    def _init_observation_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Build the observation transition matrix and noise matrix.
        The noise is computed analytically from the data, while the transition
        matrix has the form $$\begin{bmatrix}I_2 & 0_2\end{bmatrix}$$
        """
        I = torch.eye(self.obs_dim)
        Z = torch.zeros(self.obs_dim, self.obs_dim)
        H = torch.hstack((I, Z))
        R = self.approximate_covariance
        return H,R

    def _filter_init(self, values: MomentumResults, batch: int) -> MomentumResults:
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
        """Filtering step. We save the observation covariance and reset it at each step."""
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

    def _loglikelihood(self, values: MomentumResults, _: KalmanStatistics) -> torch.Tensor:
        r"""Calculate the log-likelihood for the model given the current state.

        Since Q, our latent covariance, is non-PSD, we have to use a pseudo inverse to approximate
        the log-likelihood.
        First we perform SVD where $$\bar{Q} = U\Sigma U^T$$
        We can then use the pseudo-inverse and make $\Sigma^+ = \Sigma^{-1}$, but only using the upper non-zero corner.
        $\Sigma$ contains our matrice's positive eigenvalues.

        Our pseudo-determinant then becomes $|\Sigma|^+ = \prod_{i=1}^r \sigma_i$

        We have to replace our normalization constant's $d$ value with the matrix rank $r$
        The resulting log-density for the initial and latent states becomes:
            $$-\frac{1}{2}(ln|\Sigma|^+ + rlog(2\pi) + z_t^T \Sigma^+ z_t)$$
        
        Our full log-likelihood expression is:
        $$ln\ P(X,Z|\theta) =  ln\ P(z_1) + \sum_{t=2}^T ln\ P(z_t|z_{t-1},\sigma,\lambda)
            + \sum_{t=1}^T ln\ P(x_t|z_t)$$
        Args:
            values (MomentumResults): The current decoded values for hidden states.
            _ (KalmanStatistics): Ignored for this model.

        Returns:
            torch.Tensor: The log-likelihood for the model.
        """
        loglike = 0.

        A   = self.transition_matrices
        C   = self.observation_matrices
        mu0 = self.initial_state_mean
        Sigma  = self.observation_covariance
        Gamma  = self.transition_covariance
        Gamma0 = self.initial_state_covariance

        rank = self.obs_dim
        GammaU,GammaS,GammaVh = torch.linalg.svd(Gamma)
        GammaPinv = torch.diag(1 / GammaS[:rank])
        GammaPlogdet = torch.log(torch.prod(GammaS[:rank]))

        log2pi = torch.log(2*PI)

        for b in range(len(values.observations)): 
            T = len(values.observations[b])
            _loglike = 0

            muhat = values.smoothed_mean[b]
            vhat  = values.smoothed_cov[b]
            Jhat  = values.smoothed_gain[b]
            
            Exx  = values.observations[b] @ values.observations[b].mT
            Ezz  = vhat + muhat @ muhat.mT
            Ezz1 = Jhat[:-1] @ vhat[1:] + muhat[1:] @ muhat[:-1].mT 
            
            ill_c = torch.logdet(Gamma0) + rank * log2pi
            zll_c = (T-1) * (GammaPlogdet + rank * log2pi)
            oll_c = T * (torch.sum(torch.logdet(Sigma[b])) + rank * log2pi)
            _loglike += ill_c + zll_c + oll_c

            ip1 = Ezz[0,:self.obs_dim,:self.obs_dim]
            ip2 = mu0 @ muhat[0,:self.obs_dim].mT
            ip3 = mu0 @ mu0.mT
            ill = torch.trace(hseu.mulinv(
                2*Gamma0,
                ip1 - ip2 - ip2.mT + ip3
            ))
            zp1 = Ezz[1:,:self.obs_dim,:self.obs_dim]
            zp2 = (Ezz1 @ A.mT)[:,:self.obs_dim,:self.obs_dim]
            zp3 = (A @ Ezz[:-1] @ A.mT)[:,:self.obs_dim,:self.obs_dim]
            zll = GammaPinv @ torch.sum(
                zp1 - zp2 - zp2.mT + zp3, 
                axis=0
            )
            zll = torch.trace(zll)

            op1 = Exx 
            op2 = C @ muhat @ values.observations[b].mT
            op3 = C @ Ezz @ C.mT
            oll = torch.sum(hseu.mulinv(
                Sigma[b],
                op1 - op2 - op2.mT + op3
            ), axis=0)
            oll = torch.trace(oll)

            _loglike += ill + zll + oll

            loglike -= _loglike / 2.

        return loglike

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
            optimizer (str): The optimizer to usepredicted_mean = _copy(tf_base),
            predicted_cov  = _copy(cov_base),
            filtered_mean  = _copy(tf_base),
            filtered_cov   = _copy(cov_base),
            smoothed_gain  = _copy(cov_base),
            smoothed_mean  = _copy(tf_base),
            smoothed_cov   = _copy(cov_base),.
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
            
            M     = -torch.exp(decay) * self.dt + 1 
            sigma = torch.exp(diffusion)**2 * self.dt

            total_loss = 0
            for i in range(n_batches):
                loss = 0 
                T = len(Ezz[i])

                loss = Ezz[i][1:] - 2 * M * Ezz1[i] + M**2 * Ezz[i][:-1]
                loss = torch.sum(loss, axis=0) / sigma
                loss = (T-1) * torch.log(sigma**2) + loss

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
        
        self.decay     = params[0].detach()
        self.diffusion = params[1].detach()

        (
            self.transition_matrices,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        return loss

    def fit(self, 
            X=None, 
            n_iter: int = 1000, 
            emtol: float = 1e-3, 
            checkpoint_path: str|None = None,
            **maximization_args
        ) -> MomentumResults:
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            X (None): Value ignored. We treat self.approx_mean as the observed variable.
            n_iter (int): Number of EM iterations.
            emtol (float): Tolerance for the change in log-likelihood between iterations.
            checkpoint_path (str|None): Path to save checkpoint files. Checkpoint files are deleted after a successful run.
            **maximization_args: Keyword arguments to pass to the parent class's maximization method.

        Returns:
            MomentumResults: Results of fitting the model to the data.
        """
        return super().fit(
            self.approximate_mean,
            n_iter,
            emtol,
            checkpoint_path,
            **maximization_args
        )