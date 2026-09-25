import torch
import numpy as np
import copy
import warnings
from dataclasses import dataclass, field
from torch.distributions import MultivariateNormal

import hippocampalseq.utils as hseu
from .statespace import *

__all__ = [
    'LinearGaussianSystem',
    'LDSResults',
    'LDSStatistics',
    'LDSParameters',
    'PI'
]

torch.set_default_dtype(torch.double)
PI = torch.tensor(np.pi)

@dataclass
class LDSResults:
    """Results of fitting the linear gaussian system

    :param observations            : list[torch.Tensor] : Observed values $x_t$. Real emitted data,
    :param predicted_mean          : list[torch.Tensor] : Predicted mean $\mu_{t|t-1}$ 
    :param predicted_cov           : list[torch.Tensor] : Predicted covariance $V_{t|t-1}$
    :param filtered_mean           : list[torch.Tensor] : Filtered mean $\mu_{t|t}$
    :param filtered_cov            : list[torch.Tensor] : Filterd covariance $V_{t|t}$ 
    :param smoothed_gain           : list[torch.Tensor] : RTS smoothing gain $J_t$ 
    :param smoothed_mean           : list[torch.Tensor] : Smoothed mean $\hat{\mu}_{t|T}$ 
    :param smoothed_cov            : list[torch.Tensor] : Smoothed covariance $\hat{V}_{t|T}$ 
    :param loglike                 : list[float]        : Observed log likelihood $ln\ P(x_{1...T}|\theta)$
    :param loglike_full            : torch.Tensor       : Complete data log likelihood $ln\ P(x_{1...T},z_{1...T}|\theta)$
    :param cumulative_probabilities: torch.Tensor       : Cumulative probabilities for each time-series.
    :param aic                     : float              : Akaike information criterion
    :param bic                     : float              : Bayesian information criterion 
    """
    observations             : list[torch.Tensor] = field(default_factory=list) 
    predicted_mean           : list[torch.Tensor] = field(default_factory=list) 
    predicted_cov            : list[torch.Tensor] = field(default_factory=list) 
    filtered_mean            : list[torch.Tensor] = field(default_factory=list) 
    filtered_cov             : list[torch.Tensor] = field(default_factory=list) 
    smoothed_gain            : list[torch.Tensor] = field(default_factory=list) 
    smoothed_mean            : list[torch.Tensor] = field(default_factory=list) 
    smoothed_cov             : list[torch.Tensor] = field(default_factory=list) 
    loglike                  : list[float]        = field(default_factory=list)
    loglike_full             : torch.Tensor       = field(default_factory=lambda: torch.empty(0))
    cumulative_probabilities : torch.Tensor       = field(default_factory=lambda: torch.empty(0))
    aic                      : float              = 0
    bic                      : float              = 0


@dataclass
class LDSStatistics:
    """Sufficient statistics for maximizing the parameters of the LDS.

    :param Cov : list[torch.Tensor] : $\hat{V}_tJ_{t-1}$
    :param Ez  : list[torch.Tensor] : $\mathbb{E}[z^T]$
    :param Ezz : list[torch.Tensor] : $\mathbb{E}[zz^T]$
    :param Ezz1: list[torch.Tensor] : $\mathbb{E}[z_{t}z_{t-1}^T]$
    :param Ez1z: list[torch.Tensor] : $\mathbb{E}[z_{t-1}z_t^T]$
    :param Exx : list[torch.Tensor] : $\mathbb{E}[xx^T]$
    :param Exz : list[torch.Tensor] : $\mathbb{E}[xz^T]$
    :param Ezx : list[torch.Tensor] : $\mathbb{E}[zx^T]$
    """
    Cov  : list[torch.Tensor] = field(default_factory=list) # $\hat{V}_tJ_{t-1}$
    Ez   : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[z^T]$
    Ezz  : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[zz^T]$
    Ezz1 : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[z_{t}z_{t-1}^T]$
    Ez1z : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[z_{t-1}z_t^T]$
    Exx  : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[xx^T]  $
    Exz  : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[xz^T]  $
    Ezx  : list[torch.Tensor] = field(default_factory=list) # $\mathbb{E}[zx^T]$

@dataclass
class LDSParameters:
    """Parameters of the LDS

    :param transition_matrix     : torch.Tensor : Transition matrix $F_t$
    :param transition_covariance : torch.Tensor : Transition covariance $Q_t$
    :param transition_bias       : torch.Tensor : Transition bias $b_t$
    :param emission_matrix       : torch.Tensor : Emission matrix $H_t$
    :param emission_covariance   : torch.Tensor : Emission covariance $R_t$
    :param emission_bias         : torch.Tensor : Emission bias $d_t$
    :param initial_mean          : torch.Tensor : Initial mean $\mu_0$
    :param initial_covariance    : torch.Tensor : Initial covariance $V_0$
    """
    transition_matrix     : torch.Tensor
    transition_covariance : torch.Tensor 
    transition_bias       : torch.Tensor 
    emission_matrix       : torch.Tensor
    emission_covariance   : torch.Tensor 
    emission_bias         : torch.Tensor 
    initial_mean          : torch.Tensor
    initial_covariance    : torch.Tensor

class LinearGaussianSystem(StateSpace):
    r"""Implementation of Kalman Filtering and RTS smoothing to solve for
    linear dynamic systems with Gaussian transition and emission functions.
        $$\begin{align}
            z_t &= F_tz_{t-1} + b_t + \xi_t \\
            x_t &= H_tz_t + d_t + \eta_t \\
                \xi_t &\sim \mathcal{N}(0, Q_t) \\
                \eta_t &\sim \mathcal{N}(0, R_t)
        \end{align}$$
    """
    def __init__(
        self,
        latent_dim: int, 
        emission_dim: int,
        order: int = 1,
        default_params: dict[str, torch.Tensor] = {},
        environment_size: list[tuple[int,...]]|None = None,
        bin_size: float|None = None,
        initialization_method: str|dict[str, hseu.NDArray] = 'uniform'
    ):
        r"""Initialize the LDS.

        Args:
            latent_dim (int): Dimension of latent state.
            emission_dim (int): Dimension of observation.
            order (int): Order of the LDS. The augmented state will be order * latent_dim. Default: 1
            default_params (dict[str, torch.Tensor], optional): Dictionary of default parameters.
                These parameters will not be learned. Default {}.
            environment_size (list[tuple[int,...]]|None, optional): Size of the environment the code is being run in.
                Only use to calculate the marginal distribution. Default: None. 
            bin_size (float|None, optional): Bin size in centimeters. Only use to calculate the marginal distribution. Default: None
            initialization_method (str|dict[str, hseu.NDArray], optional): Method for initializing the parameters.
                Either a string ['normal', 'uniform'] or a dict with named parameters. Default: 'uniform'
        """

        self.latent_dim = latent_dim
        self.augmented_dim = order * latent_dim
        self.emission_dim = emission_dim

        self.no_em_vars = list(default_params.keys())
        self.default_parameters = copy.deepcopy(default_params)

        # Sum of all matrix sizes minus those that are default
        self.n_parameters = 3 * (self.augmented_dim * self.augmented_dim) \
            + 2 * self.augmented_dim \
            + self.emission_dim * self.augmented_dim \
            + self.emission_dim * self.emission_dim \
            + self.emission_dim \
            - sum([p.numel() for p in default_params.values()])

        self.environment_size = environment_size
        self.bin_size = bin_size
        self.initialization_method = initialization_method

    def name(self):
        return "Linear Gaussian Dynamic System"

    def random_initializer(self, parameter_name: str, shape: tuple[int, ...] | int, method: str|dict) -> torch.Tensor:
        if isinstance(method, dict):
            return hseu.ensure_torch(method[parameter_name])
        return hseu.ensure_torch(super().random_initializer(shape, method))
        
    def _initialize_observations(self, X: torch.Tensor|list[torch.Tensor]|None):
        if X is None:
            raise ValueError("Observation data cannot be None")
        if not isinstance(X, list):
            X = [X]
        for i in range(len(X)):
            assert X[i] is not None, f"Observation {i} is None"
            X[i] = hseu.ensure_torch(hseu.atleast_3d(X[i]))
        return X

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        F = self.random_initializer(
            "transition_matrix",
            (self.augmented_dim, self.augmented_dim), 
            self.initialization_method
        )
        return F / F.sum(axis=1, keepdim=True)
    
    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        Q = self.random_initializer(
            "transition_covariance",
            (self.augmented_dim, self.augmented_dim), 
            self.initialization_method
        )
        return Q @ Q.T 
    
    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        return self.random_initializer(
            "transition_bias",
            (self.augmented_dim, 1), 
            self.initialization_method
        )

    def _construct_emission_matrix(self, mode: str = "train") -> torch.Tensor:
        H = self.random_initializer(
            "emission_matrix",
            (self.emission_dim, self.augmented_dim), 
            self.initialization_method
        )
        return H / H.sum(axis=1, keepdim=True)
    
    def _construct_emission_covariance(self, mode: str = "train") -> torch.Tensor:
        R = self.random_initializer(
            "emission_covariance",
            (self.emission_dim, self.emission_dim), 
            self.initialization_method
        )
        return R @ R.T
    
    def _construct_emission_bias(self, mode: str = "train") -> torch.Tensor:
        return self.random_initializer(
            "emission_bias",
            (self.emission_dim, 1), 
            self.initialization_method
        )
    
    def _construct_initial_mean(self, mode: str = "train") -> torch.Tensor:
        return self.random_initializer(
            "initial_mean",
            (self.augmented_dim, 1), 
            self.initialization_method
        )
    
    def _construct_initial_covariance(self, mode: str = "train") -> torch.Tensor:
        return torch.eye(self.augmented_dim)

    def _initialize_globals(self, mode: str = "train"):
        default = lambda name, value: self.default_parameters.get(name, value)

        params = LDSParameters(
            transition_matrix = default("transition_matrix", 
                self._construct_transition_matrix(mode)
            ),
            transition_covariance = default("transition_covariance",
                    self._construct_transition_covariance(mode)
            ),
            transition_bias = default("transition_bias",
                self._construct_transition_bias(mode)
            ),
            emission_matrix = default("emission_matrix",
                self._construct_emission_matrix(mode)
            ),
            emission_covariance = default("emission_covariance",
                self._construct_emission_covariance(mode),
            ),
            emission_bias = default("emission_bias",
                self._construct_emission_bias(mode)
            ),
            initial_mean = default("initial_mean", 
                self._construct_initial_mean(mode)
            ),
            initial_covariance = default("initial_covariance",
                self._construct_initial_covariance(mode)
            )
        )
        self.global_parameters = params
    
    def _initialize_values(self, X: list[torch.Tensor]) -> LDSResults:
        meanbase = lambda: [torch.zeros((len(x), self.augmented_dim, 1)) for x in X]
        covbase = lambda: [torch.zeros((len(x), self.augmented_dim, self.augmented_dim)) for x in X]
        return LDSResults(
            observations   = X,
            predicted_mean = meanbase(), 
            predicted_cov  = covbase(),
            filtered_mean  = meanbase(),
            filtered_cov   = covbase(),
            smoothed_gain  = covbase(),
            smoothed_mean  = meanbase(),
            smoothed_cov   = covbase()
        )

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        """Build parameters for a specific batch.
        
        Args:
            batch: Batch index
            mode (str): Either "train", "valid", or "test" to differentiate training, validation, and testing data.
            
        Returns:
            (LDSParameters): Parameters for this batch
        """
        assert mode in ["train", "valid", "test"], "Mode must be train, valid, or test"
        return self.global_parameters

    def filter(self, values: LDSResults, mode: str = "train") -> LDSResults:
        """Run the Kalman Filter.
        
        Args:
            values: LDSResults with observations
            mode (str): Either "train", "valid", or "test".
        """
        for batch in range(len(values.observations)):
            batch_params = self.build_batch_parameters(batch, mode=mode)
            values = self._filter_init(values, batch_params, batch)
            for t in range(1, len(values.observations[batch])):
                values = self._filter(values, batch_params, batch, t)
        return values

    def smooth(self, values: LDSResults, mode: str = "train") -> LDSResults:
        """Run the RTS smoother.
        
        Args:
            values: LDSResults with filtered values
            mode (str): Either "train", "valid", or "test".
        """
        for batch in range(len(values.observations)):
            batch_params = self.build_batch_parameters(batch, mode=mode)
            values = self._smooth_init(values, batch_params, batch)
            for t in reversed(range(len(values.observations[batch]) - 1)):
                values = self._smooth(values, batch_params, batch, t)
        return values

    def _filter_init(self, values: LDSResults, batch_params: LDSParameters, batch: int) -> LDSResults:
        H   = hseu.extract_last_dims(batch_params.emission_matrix, 0)
        R   = hseu.extract_last_dims(batch_params.emission_covariance, 0)
        d   = hseu.extract_last_dims(batch_params.emission_bias, 0)
        x0  = hseu.extract_last_dims(values.observations[batch], 0)
        mu0 = batch_params.initial_mean
        v0  = batch_params.initial_covariance

        if torch.any(x0.isnan()):
            mu1  = mu0
            v1   = v0
        else:
            P0Ct = v0 @ H.T
            K1 = hseu.invmul(P0Ct, H @ P0Ct + R)
            innovation = x0 - H @ mu0 - d
            mu1 = mu0 + K1 @ innovation
            v1 = (torch.eye(self.augmented_dim) - K1 @ H) @ v0

        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = v1
        values.predicted_mean[batch][0] = mu0
        values.predicted_cov[batch][0]  = v0
        return values

    def _filter(self, values: LDSResults, batch_params: LDSParameters, batch: int, t: int) -> LDSResults:
        H    = hseu.extract_last_dims(batch_params.emission_matrix, t)
        R    = hseu.extract_last_dims(batch_params.emission_covariance, t)
        d    = hseu.extract_last_dims(batch_params.emission_bias, t)
        F    = hseu.extract_last_dims(batch_params.transition_matrix, t)
        Q    = hseu.extract_last_dims(batch_params.transition_covariance, t)
        b    = hseu.extract_last_dims(batch_params.transition_bias, t)
        xt   = hseu.extract_last_dims(values.observations[batch], t)
        mut1 = values.filtered_mean[batch][t-1]
        vt1  = values.filtered_cov[batch][t-1]

        # $\mu_{t|t-1} = F \mu_{t-1|t-1} + b$
        # $$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
        Am1 = F @ mut1 + b
        Pn1 = F @ vt1 @ F.T + Q

        if torch.any(xt.isnan()):
            mut = Am1
            vt  = Pn1
        else:
            # $K = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$
            Pct = Pn1 @ H.T
            K = hseu.invmul(Pct, H @ Pct + R)
            # $\mu_{t|t} = \mu_{t|t-1} + K (x_t - H \mu_{t|t-1} - d)$
            # $$P_{t|t} = (I - K H) P_{t|t-1}$$
            innovation = xt - H @ Am1 - d
            mut = Am1 + K @ innovation 
            vt  = (torch.eye(self.augmented_dim) - K @ H) @ Pn1

        values.filtered_mean[batch][t]  = mut # $\mu_{t|t}$
        values.filtered_cov[batch][t]   = vt  # $P_{t|t}$
        values.predicted_mean[batch][t] = Am1 # $\mu_{t|t-1}$
        values.predicted_cov[batch][t]  = Pn1 # $P_{t|t-1}$
        return values

    def _smooth_init(self, values: LDSResults, _: LDSParameters, batch: int) -> LDSResults:
        # $\hat{\mu}_T = \mu_{T|T}$
        # $\hat{P}_T = P_{T|T}$
        values.smoothed_mean[batch][-1] = values.filtered_mean[batch][-1]
        values.smoothed_cov[batch][-1]  = values.filtered_cov[batch][-1]
        return values

    def _smooth(self, values: LDSResults, batch_params: LDSParameters, batch: int, t: int) -> LDSResults:
        F = hseu.extract_last_dims(batch_params.transition_matrix, t)
        Amt = values.predicted_mean[batch][t+1] # $\mu_{t+1|t}$
        Pt  = values.predicted_cov[batch][t+1]  # $P_{t+1|t}$
        mt  = values.filtered_mean[batch][t]  # $\mu_{t|t}$
        vt  = values.filtered_cov[batch][t]   # $P_{t|t}$

        # $J_t = P_{t|t}F^T P_{t+1|t}^{-1}$
        J = hseu.invmul(vt @ F.T, Pt)
        # $\hat{\mu}_t = \mu_{t|t} + J_t ( \hat{\mu}_{t+1} - \mu_{t+1|t} )$
        # $$\hat{P}_t = P_{t|t} + J_t ( \hat{P}_{t+1} - P_{t+1|t} ) J_t^T$$
        muht = mt + J @ (values.smoothed_mean[batch][t+1] - Amt)
        vht  = vt + J @ (values.smoothed_cov[batch][t+1] - Pt) @ J.mT

        values.smoothed_gain[batch][t] = J
        values.smoothed_mean[batch][t] = muht
        values.smoothed_cov[batch][t]  = vht
        return values
        
    def _observed_loglikelihood(self, values: LDSResults, mode: str = "train") -> torch.Tensor:
        r"""Calculates the observed log-likelihood of the data in the Kalman filter.
        Use this when checking for EM convergence.
        The formula is:
        $$
            L(\theta) = ln\ P(x_{1...T}|\theta) = \sum_{t=1}^T ln\ P(x_t|x_{t-1}\theta) = 
            -\frac{1}{2} \sum_{t=1}^T m ln\ (2\pi) + ln\ |S_t| + \nu_t^TS_t^{-1}\nu_t
        $$
        where $S_t = C_t P_{t|t-1} C_t^T + \Gamma$ and $\nu_t = x_t - C\mu_{t|t-1}$

        """
        log2pi = torch.log(2 * PI)
        rank = self.augmented_dim 
        loglike = torch.tensor([[0.0]])

        for b in range(len(values.observations)):
            params = self.build_batch_parameters(b, mode=mode)
            T = len(values.observations[b])
            innovation = values.observations[b] \
                - params.emission_matrix @ values.predicted_mean[b] \
                - params.emission_bias
            innovation_cov = params.emission_matrix @ values.predicted_cov[b] @ params.emission_matrix.mT \
                + params.emission_covariance

            L = torch.linalg.cholesky(innovation_cov)
            alpha = torch.cholesky_solve(innovation, L)

            loglike += T * rank * log2pi \
                + 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1))) \
                + torch.sum(innovation.mT @ alpha, axis=0)

        return -0.5 * loglike.squeeze()

    def _complete_loglikelihood(self, values: LDSResults, stats: LDSStatistics, mode: str = "train") -> torch.Tensor:
        """Calculate the complete data log likelihood of the model given the sufficient statistics and current 
        parameters.

        Args:
            values (LDSResults): The filtered and smoothed values of the model.
            stats (LDSStatistics): The sufficient statistics of the model.
            mode (str): Either "train", "valid", or "test"

        Returns:
            torch.Tensor: The log likelihood of the model.
        """
        log2pi = torch.log(2 * PI)
        rank = self.augmented_dim
        loglike = torch.tensor([[0.0]])

        for b in range(len(values.observations)):
            params = self.build_batch_parameters(b, mode=mode)
            T = len(values.observations[b])
            t = np.arange(T)
            tmat = hseu.extract_last_dims(params.transition_matrix, t[1:])
            tcov = hseu.extract_last_dims(params.transition_covariance, t[1:])
            emat = hseu.extract_last_dims(params.emission_matrix, t)
            ecov = hseu.extract_last_dims(params.emission_covariance, t)
            tbias = hseu.extract_last_dims(params.transition_bias, t)
            ebias = hseu.extract_last_dims(params.emission_bias, t)

            iloglike = 0
            
            # Log-determinant of the Gaussian portions
            ilogd = torch.logdet(params.initial_covariance)
            tlogd = torch.logdet(tcov)
            if tcov.ndim == 2:
                tlogd *= (T - 1)
            else:
                tlogd = torch.sum(tlogd, axis=0)
            elogd = torch.logdet(ecov)
            if ecov.ndim == 2:
                elogd *= T 
            else:
                elogd = torch.sum(elogd, axis=0)
            iloglike += ilogd + tlogd + elogd

            # Initial state 
            ip1 = stats.Ezz[b][0]
            ip2 = params.initial_mean @ values.smoothed_mean[b][0].mT 
            ip3 = params.initial_mean @ params.initial_mean.mT 
            ip  = ip1 - ip2 - ip2.mT + ip3
            # Initial state bias term
            tbias0 = tbias[0] if tbias.ndim == 3 else tbias
            ipb1 = stats.Ez[b][0] @ tbias0.mT
            ipb2 = params.initial_mean @ tbias0.mT
            ipb3 = tbias0 @ tbias0.mT
            ipb = ipb3 + ipb2 + ipb2.mT - ipb1 - ipb1.mT
            # Initial loglike
            ill = hseu.invmul(params.initial_covariance, ip + ipb)
            iloglike += torch.trace(ill)

            # Base transition term 
            tp1 = stats.Ezz[b][1:]
            tp2 = stats.Ezz1[b] @ tmat.mT 
            tp3 = tmat @ stats.Ezz[b][:-1] @ tmat.mT 
            tp  = tp1 - tp2 - tp2.mT + tp3
            # Transition bias term 
            bias = tbias[1:] if tbias.ndim == 3 else tbias
            tpb1 = stats.Ez[b][1:] @ bias.mT 
            tpb2 = tmat @ stats.Ez[b][:-1] @ bias.mT 
            tpb3 = bias @ bias.mT 
            tpb  = tpb3 + tpb2 + tpb2.mT - tpb1 - tpb1.mT 
            # Transition loglike
            if tcov.ndim == 2:
                tll = hseu.mulinv(tcov, torch.sum(tp + tpb, axis=0))
            else:
                tll = hseu.mulinv(tcov, tp + tpb)
                tll = torch.sum(tll, axis=0)
            iloglike += torch.trace(tll)

            # Base emission term
            ep1 = stats.Exx[b]
            ep2 = stats.Exz[b] @ emat.mT 
            ep3 = emat @ stats.Ezz[b] @ emat.mT 
            ep  = ep1 - ep2 - ep2.mT + ep3
            # Emission bias term 
            epb1 = values.observations[b] @ ebias.mT 
            epb2 = emat @ stats.Ez[b] @ ebias.mT 
            epb3 = ebias @ ebias.mT 
            epb  = epb3 + epb2 + epb2.mT - epb1 - epb1.mT 
            # Emission loglike
            if ecov.ndim == 2:
                ell = hseu.mulinv(ecov, torch.sum(ep + epb, axis=0))
            else:
                ell = hseu.mulinv(ecov, ep + epb)
                ell = torch.sum(ell, axis=0)
            iloglike += torch.trace(ell)

            loglike += iloglike + T * rank * log2pi

        return -0.5 * loglike.squeeze()

    def _calculate_sufficient_statistics(self, values: LDSResults) -> LDSStatistics:
        """Calculate sufficient statistics for performing maximization given the filtered
         and smoothed values of the model.

        Args:
            values (LDSResults): The filtered and smoothed values of the model.

        Returns:
            LDSStatistics: The sufficient statistics of the model.
        """
        stats = LDSStatistics()
        for b in range(len(values.observations)):
            cov  = values.smoothed_cov[b][1:] @ values.smoothed_gain[b][:-1].mT 
            ez   = values.smoothed_mean[b] 
            ezz  = ez @ ez.mT + values.smoothed_cov[b] 
            ezz1 = values.smoothed_mean[b][1:] @ values.smoothed_mean[b][:-1].mT + cov
            ez1z = ezz1.mT 
            exx  = values.observations[b] @ values.observations[b].mT 
            exz  = values.observations[b] @ values.smoothed_mean[b].mT 
            ezx  = exz.mT
            stats.Cov.append(cov)
            stats.Ez.append(ez)
            stats.Ezz.append(ezz)
            stats.Ezz1.append(ezz1)
            stats.Ez1z.append(ez1z)
            stats.Exx.append(exx)
            stats.Exz.append(exz)
            stats.Ezx.append(ezx)
        return stats

    def _initial_mean_mle(self, stats: LDSStatistics) -> torch.Tensor:
        return hseu.atleast_2d(
            torch.mean(torch.cat([
                sm[0].unsqueeze(0) for sm in stats.Ez
            ]), axis=0)
        )

    def _initial_cov_mle(self, stats: LDSStatistics) -> torch.Tensor:
        P1 = torch.cat([ezz[0].unsqueeze(0) for ezz in stats.Ezz])
        P2 = torch.cat([(ez[0] @ ez[0].mT).unsqueeze(0) for ez in stats.Ez])
        return hseu.atleast_2d(torch.mean(P1 - P2, axis=0))

    def _transition_matrix_mle(self, stats: LDSStatistics) -> torch.Tensor:
        Numer = torch.cat([torch.sum(ezz1, axis=0, keepdim=True) for ezz1 in stats.Ezz1])
        Denom = torch.cat([torch.sum(ezz, axis=0, keepdim=True) for ezz in stats.Ezz])
        A = hseu.invmul(Numer, Denom)
        return hseu.atleast_2d(torch.mean(A, axis=0))

    def _transition_cov_mle(self, stats: LDSStatistics) -> torch.Tensor:
        P1 = [ezz[1:] for ezz in stats.Ezz]
        P2 = [ezz1 @ self.global_parameters.transition_matrix.T for ezz1 in stats.Ezz1]
        P3 = [p2.mT for p2 in P2]
        P4 = [
            self.global_parameters.transition_matrix @ ezz[:-1] @ self.global_parameters.transition_matrix.mT
            for ezz in stats.Ezz
        ]
        Gamma = torch.cat([torch.sum(p1-p2-p3+p4, axis=0, keepdim=True) / len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return hseu.atleast_2d(torch.mean(Gamma, axis=0))

    def _emission_matrix_mle(self, stats: LDSStatistics) -> torch.Tensor:
        Numer = torch.cat([torch.sum(exz, axis=0, keepdim=True) for exz in stats.Exz])
        Denom = torch.cat([torch.sum(ezz, axis=0, keepdim=True) for ezz in stats.Ezz])
        C = hseu.invmul(Numer, Denom)
        return hseu.atleast_2d(torch.mean(C, axis=0))

    def _emission_cov_mle(self, stats: LDSStatistics) -> torch.Tensor:
        P1 = stats.Exx
        P2 = [self.global_parameters.emission_matrix @ ezx for ezx in stats.Ezx]
        P3 = [p2.mT for p2 in P2]
        P4 = [
            self.global_parameters.emission_matrix @ ezz @ self.global_parameters.emission_matrix.mT
            for ezz in stats.Ezz
        ]
        Sigma = torch.cat([torch.sum(p1-p2-p3+p4, axis=0, keepdim=True)/len(p1) for p1,p2,p3,p4 in zip(P1, P2, P3, P4)])
        return hseu.atleast_2d(torch.mean(Sigma, axis=0))

    def _transition_bias_mle(self, values: LDSResults) -> torch.Tensor:
        r"""Transition bias MLE.

        $b = \frac{1}{T-1}\sum_{t=2}^T \mathbb{E}[z_t] - A \mathbb{E}[z_{t-1}]$

        """

        offset = torch.cat([
            (ez[1:] - self.global_parameters.emission_matrix @ ez[:-1])/(len(ez) - 1) 
            for ez in values.smoothed_mean
        ])
        offset = torch.mean(offset, axis=0)
        return hseu.atleast_2d(offset)


    def _emission_bias_mle(self, values: LDSResults) -> torch.Tensor:
        r"""Observation bias MLE.

        $d = \frac{1}{T}\sum_{t=1}^T \mathbb{E}[x_t] - C \mathbb{E}[z_t]$
        """

        offset = torch.cat([
            (obs - self.global_parameters.emission_matrix @ sm) / len(obs)
            for obs, sm in zip(values.observations, values.smoothed_mean)
        ])
        offset = torch.mean(offset, axis=0)
        return hseu.atleast_2d(offset)

    def _solve_parameters(self, values: LDSResults, stats: LDSStatistics, **_):
        with torch.no_grad():
            if "transition_matrix" not in self.no_em_vars:
                self.global_parameters.transition_matrix = self._transition_matrix_mle(stats)
            if "transition_cov" not in self.no_em_vars:
                self.global_parameters.transition_covariance = self._transition_cov_mle(stats)
            if "transition_bias" not in self.no_em_vars:
                self.global_parameters.transition_bias = self._transition_bias_mle(values)
            if "emission_matrix" not in self.no_em_vars:
                self.global_parameters.emission_matrix = self._emission_matrix_mle(stats)
            if "emission_cov" not in self.no_em_vars:
                self.global_parameters.emission_covariance = self._emission_cov_mle(stats)
            if "emission_bias" not in self.no_em_vars:
                self.global_parameters.emission_bias = self._emission_bias_mle(values)
            if "initial_mean" not in self.no_em_vars:
                self.global_parameters.initial_mean = self._initial_mean_mle(stats)
            if "initial_cov" not in self.no_em_vars:
                self.global_parameters.initial_covariance = self._initial_cov_mle(stats)

    def _calculate_marginals(self, environment_size: list[tuple[int,...]], bin_size: int, values: KalmanResults) -> torch.Tensor:
        r"""Calculates the marginal probabilities for each bin in the environment.
        What is the probability that the mouse is in a given bin at a given time $P(X_t = x, Y_t = y|\mu_t, \Sigma_t)$

        Args:
            environment_size (list[tuple[int,...]]): List of tuples of size (min,max) for each axis.
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
            sm = values.smoothed_mean[i][:,self.latent_dim:]
            sc = torch.atleast_2d(values.smoothed_cov[i][:,self.latent_dim:,self.latent_dim:])
            cp = torch.zeros((sm.shape[0],)+sz)
            for t in range(sm.shape[0]):
                cov_t = sc[t]
                U, S, Vh = torch.linalg.svd(cov_t)
                S_clamped = torch.clamp(S, min=1e-8)
                cov_reg = U @ torch.diag(S_clamped) @ Vh
                try:
                    L = torch.linalg.cholesky(cov_reg)
                except Exception as e:
                    print(i,t)
                    print(cov_reg)
                    print(S)
                    print(cov_t)
                    print(values.smoothed_cov[i][t])
                    raise e

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

    def e_step(self, values: LDSResults, mode: str = "train") -> tuple[LDSResults, torch.Tensor]:
        """E-step: filter and smooth, then compute log-likelihood.
        
        Args:
            values (LDSResults): LDSResults with observations
            mode (str): Either "train", "valid", or "test".
            
        Returns:
            (LDSResults): Filtered and smoothed values from the model.
            (torch.Tensor): Observed log-likelihood calculated from the filtered values.
        """
        with torch.no_grad():
            values = self.filter(values, mode=mode)
            values = self.smooth(values, mode=mode)
            ll = self._observed_loglikelihood(values, mode=mode)
        return (values, ll)

    def m_step(self, values: KalmanResults, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            stats = self._calculate_sufficient_statistics(values)
        return self._solve_parameters(values, stats, **kwargs)

    def fit(self,
            Xtrain: list[hseu.NDArray],
            Xvalid: list[hseu.NDArray]|None=None,
            n_iter: int = 1000,
            emtol: float = 1e-3,
            patience: int = 5,
            val_freq: int = 10,
            **maximization_args
        ) -> LDSResults:
        """Fit the model with optional early stopping.
        
        Args:
            Xtrain (list[hseu.NDArray]): Training observations
            Xvalid (list[hseu.NDArray]|None): Validation observations (optional). If provided, enables early stopping. Defaults to None.
            n_iter (int): Maximum number of EM iterations. Defaults to 1000.
            emtol (float): Tolerance for convergence on training log-likelihood. Defaults to 1e-3.
            patience (int): Number of validation iterations without improvement before stopping. Only used if Xvalid is provided. Defaults to 5.
            val_freq (int): Frequency of validation checks (every val_freq iterations). Only used if Xvalid is provided. Defaults to 10.
            **maximization_args: Additional arguments for m_step
            
        Returns:
            LDSResults: Fitted results on training data
        """
        Xtrain = self._initialize_observations(Xtrain)
        self._initialize_globals("train")
        values = self._initialize_values(Xtrain)
        
        best_params = None
        best_valid_ll = -np.inf
        patience_counter = 0
        
        if Xvalid is not None:
            Xvalid = self._initialize_observations(Xvalid)
            valid_values = self._initialize_values(Xvalid)

        for i in range(n_iter):
            # E-step and M-step on training data
            values, ll = self.e_step(values, mode="train")
            self.m_step(values, **maximization_args)
            values.loglike.append(ll)

            if not torch.isfinite(values.loglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            # Convergence check on training data
            if i > 0 and abs((values.loglike[-1] - values.loglike[-2]) / values.loglike[-2]) < emtol:
                print(f"Training converged after {i} epochs, exiting")
                break

            if i % 20 == 0:
                print(f"Iteration {i}: Training loglike={ll.item():.4f}")

            # Validation step with early stopping
            if Xvalid is not None and i % val_freq == 0:
                self._initialize_globals("valid")
                valid_values, valid_ll = self.e_step(valid_values, mode="valid")
                self._initialize_globals("train")
                valid_values.loglike.append(valid_ll)
                
                if not torch.isfinite(valid_ll):
                    print(f" | Validation LL is NaN/Inf, stopping at iter {i}")
                    break
                print(f" | Validation loglike={valid_ll.item():.4f}")
                
                if valid_ll > best_valid_ll:
                    best_valid_ll = valid_ll
                    patience_counter = 0
                    best_params = copy.deepcopy(self.global_parameters)
                else:
                    patience_counter += 1
                    print(f" [patience {patience_counter}/{patience}]")
                    if patience_counter >= patience:
                        print(f"\nEarly stopping: validation LL did not improve for {patience} checks")
                        if best_params is not None:
                            self.global_parameters = best_params
                        break
            
        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        if best_params is not None:
            self.global_parameters = best_params

        values = self.e_step(values, mode="train")[0]
        values.loglike_full = self._complete_loglikelihood(
            values,
            self._calculate_sufficient_statistics(values),
            mode="train"
        )
        values.aic = self.aic(values.loglike_full)
        values.bic = self.bic(
            values.loglike_full,
            sum(len(obs) for obs in values.observations)
        )

        if self.environment_size is not None and self.bin_size is not None:
            values.cumulative_probabilities = self._calculate_marginals(
                self.environment_size, 
                self.bin_size, 
                values
            )

        if Xvalid is None:
            return values

        self._initialize_globals("valid")
        valid_values = self.e_step(valid_values, mode="valid")[0]
        if self.environment_size is not None and self.bin_size is not None:
            valid_values.cumulative_probabilities = self._calculate_marginals(
                self.environment_size,
                self.bin_size,
                valid_values
            )
        return values, valid_values

    def transform(
            self,
            X: list[hseu.NDArray]
        ) -> LDSResults:
        """Apply the learned parameters to unseen observations.
        Args:
            X (list[hseu.NDArray]): Observed values to run the filtering and smoothing steps over.

        Results:
            (LDSResults): Filtered and smoothed values.
        """
        X = self._initialize_observations(X)
        self._initialize_globals("test")
        values = self._initialize_values(X)
        values = self.e_step(values, mode="test")[0]
        if self.environment_size is not None and self.bin_size is not None:
            values.cumulative_probabilities = self._calculate_marginals(
                self.environment_size, 
                self.bin_size, 
                values    
            )
        return values