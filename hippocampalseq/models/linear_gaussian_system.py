import torch
import numpy as np
import copy
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
    observations             : list[torch.Tensor] = field(default_factory=list) 
    predicted_mean           : list[torch.Tensor] = field(default_factory=list) # Stores $\mu_{t+1|t}$
    predicted_cov            : list[torch.Tensor] = field(default_factory=list) # Stores $P_{t+1|t}$
    filtered_mean            : list[torch.Tensor] = field(default_factory=list) # Stores $\mu_{t|t}$
    filtered_cov             : list[torch.Tensor] = field(default_factory=list) # Stores $V_{t|t}$
    smoothed_gain            : list[torch.Tensor] = field(default_factory=list) # Stores $J_t$
    smoothed_mean            : list[torch.Tensor] = field(default_factory=list) # Stores $\hat{\mu}_{t|T}$
    smoothed_cov             : list[torch.Tensor] = field(default_factory=list) # Stores $\hat{V}_{t|T}$
    loglike                  : list[float]        = field(default_factory=list)
    loglike_full             : torch.Tensor       = field(default_factory=lambda: torch.empty(0))
    cumulative_probabilities : torch.Tensor       = field(default_factory=lambda: torch.empty(0))
    aic                      : float = 0
    bic                      : float = 0


@dataclass
class LDSStatistics:
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
    transition_matrix     : torch.Tensor
    transition_covariance : torch.Tensor 
    transition_bias       : torch.Tensor 
    emission_matrix       : torch.Tensor
    emission_covariance   : torch.Tensor 
    emission_bias         : torch.Tensor 
    initial_mean          : torch.Tensor
    initial_covariance    : torch.Tensor

class LinearGaussianSystem(StateSpace):
    def __init__(
        self,
        latent_dim: int, 
        emission_dim: int,
        order: int = 1,
        default_params: dict[str, torch.Tensor] = {},
        environment_size: list[tuple[int,...]]|None = None,
        bin_size: float|None = None
    ):

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

    def _initialize_observations(self, X: torch.Tensor|list[torch.Tensor]|None):
        if X is None:
            raise ValueError("Observation data cannot be None")
        if not isinstance(X, list):
            X = [X]
        for i in range(len(X)):
            assert X[i] is not None, f"Observation {i} is None"
            if not torch.is_tensor(X[i]):
                X[i] = torch.from_numpy(X[i])
            X[i] = hseu.atleast_3d(X[i])
        return X

    def _construct_transition_matrix(self):
        F = torch.rand(self.augmented_dim, self.augmented_dim)
        return F / F.sum(axis=1, keepdim=True)
    
    def _construct_transition_covariance(self):
        Q = torch.randn(self.augmented_dim, self.augmented_dim)
        return Q @ Q.T 
    
    def _construct_transition_bias(self):
        return torch.rand(self.augmented_dim, 1)

    def _construct_emission_matrix(self):
        return self._construct_transition_matrix()
    
    def _construct_emission_covariance(self):
        return self._construct_transition_covariance()
    
    def _construct_emission_bias(self):
        return self._construct_transition_bias()
    
    def _construct_initial_mean(self):
        return torch.randn(self.augmented_dim, 1)
    
    def _construct_initial_covariance(self):
        return torch.eye(self.augmented_dim)

    def _initialize_globals(self):
        default = lambda name, value: self.default_parameters.get(name, value)

        params = LDSParameters(
            transition_matrix = default("transition_matrix", 
                self._construct_transition_matrix()
            ),
            transition_covariance = default("transition_covariance",
                    self._construct_transition_covariance()    
            ),
            transition_bias = default("transition_bias",
                self._construct_transition_bias()
            ),
            emission_matrix = default("emission_matrix",
                self._construct_emission_matrix()
            ),
            emission_covariance = default("emission_covariance",
                self._construct_emission_covariance(),
            ),
            emission_bias = default("emission_bias",
                self._construct_emission_bias()
            ),
            initial_mean = default("initial_mean", 
                self._construct_initial_mean()
            ),
            initial_covariance = default("initial_covariance",
                self._construct_initial_covariance()
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

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        return self.global_parameters

    def filter(self, values: LDSResults) -> LDSResults:
        """Run the Kalman Filter."""
        for batch in range(len(values.observations)):
            batch_params = self.build_batch_parameters(batch)
            values = self._filter_init(values, batch_params, batch)
            for t in range(1, len(values.observations[batch])):
                values = self._filter(values, batch_params, batch, t)
        return values

    def smooth(self, values: LDSResults) -> LDSResults:
        """Run the RTS smoother."""
        for batch in range(len(values.observations)):
            batch_params = self.build_batch_parameters(batch)
            values = self._smooth_init(values, batch_params, batch)
            for t in reversed(range(len(values.observations[batch]) - 1)):
                values = self._smooth(values, batch_params, batch, t)
        return values

    def _filter_init(self, values: LDSResults, batch_params: LDSParameters, batch: int) -> LDSResults:
        H = hseu.extract_last_dims(batch_params.emission_matrix, 0)
        R = hseu.extract_last_dims(batch_params.emission_covariance, 0)
        d = hseu.extract_last_dims(batch_params.emission_bias, 0)
        F = hseu.extract_last_dims(batch_params.transition_matrix, 0)
        Q = hseu.extract_last_dims(batch_params.transition_covariance, 0)
        b = hseu.extract_last_dims(batch_params.transition_bias, 0)
        x0 = hseu.extract_last_dims(values.observations[batch], 0)
        
        P0Ct = batch_params.initial_covariance @ H.T
        K1 = hseu.invmul(P0Ct, H @ P0Ct + R)
        innovation = x0 - H @ batch_params.initial_mean - d
        mu1 = batch_params.initial_mean + K1 @ innovation
        v1 = (torch.eye(self.augmented_dim) - K1 @ H) @ batch_params.initial_covariance

        values.filtered_mean[batch][0]  = mu1
        values.filtered_cov[batch][0]   = v1
        values.predicted_mean[batch][0] = F @ mu1 + b
        values.predicted_cov[batch][0]  = F @ v1 @ F.T + Q
        return values

    def _filter(self, values: LDSResults, batch_params: LDSParameters, batch: int, t: int) -> LDSResults:
        H = hseu.extract_last_dims(batch_params.emission_matrix, t)
        R = hseu.extract_last_dims(batch_params.emission_covariance, t)
        d = hseu.extract_last_dims(batch_params.emission_bias, t)
        F = hseu.extract_last_dims(batch_params.transition_matrix, t)
        Q = hseu.extract_last_dims(batch_params.transition_covariance, t)
        b = hseu.extract_last_dims(batch_params.transition_bias, t)
        xt = hseu.extract_last_dims(values.observations[batch], t)

        Am1 = values.predicted_mean[batch][t-1]
        Pn1 = values.predicted_cov[batch][t-1]

        Pct = Pn1 @ H.T
        K = hseu.invmul(Pct, H @ Pct + R)
        innovation = xt - H @ Am1 - d
        mut = Am1 + K @ innovation 
        vt  = (torch.eye(self.augmented_dim) - K @ H) @ Pn1

        Am = F @ mut + b
        Pn = F @ vt @ F.T + Q 

        values.filtered_mean[batch][t]  = mut
        values.filtered_cov[batch][t]   = vt 
        values.predicted_mean[batch][t] = Am
        values.predicted_cov[batch][t]  = Pn
        return values

    def _smooth_init(self, values: LDSResults, _: LDSParameters, batch: int) -> LDSResults:
        values.smoothed_mean[batch][-1] = values.filtered_mean[batch][-1]
        values.smoothed_cov[batch][-1]  = values.filtered_cov[batch][-1]
        return values

    def _smooth(self, values: LDSResults, batch_params: LDSParameters, batch: int, t: int) -> LDSResults:
        F = hseu.extract_last_dims(batch_params.transition_matrix, t)
        Amt = values.predicted_mean[batch][t]
        Pt  = values.predicted_cov[batch][t]
        mt  = values.filtered_mean[batch][t]
        vt  = values.filtered_cov[batch][t]

        J = hseu.invmul(vt @ F.T, Pt)
        muht = mt + J @ (values.smoothed_mean[batch][t+1] - Amt)
        vht  = vt + J @ (values.smoothed_cov[batch][t+1] - Pt) @ J.mT

        values.smoothed_gain[batch][t] = J
        values.smoothed_mean[batch][t] = muht
        values.smoothed_cov[batch][t]  = vht
        return values
        
    def _observed_loglikelihood(self, values: LDSResults) -> torch.Tensor:
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
        loglike = 0

        for b in range(len(values.observations)):
            params = self.build_batch_parameters(b)
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

    def _complete_loglikelihood(self, values: LDSResults, stats: LDSStatistics) -> torch.Tensor:
        """Calculate the complete data log likelihood of the model given the sufficient statistics and current 
        parameters.

        Args:
            values (LDSResults): The filtered and smoothed values of the model.
            stats (LDSStatistics): The sufficient statistics of the model.

        Returns:
            torch.Tensor: The log likelihood of the model.
        """
        log2pi = torch.log(2 * PI)
        rank = self.augmented_dim
        loglike = 0

        for b in range(len(values.observations)):
            params = self.build_batch_parameters(b)
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
            sm = values.smoothed_mean[i][:,self.latent_dim:]
            sc = torch.atleast_2d(values.smoothed_cov[i][:,self.latent_dim:,self.latent_dim:])
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

    def e_step(self, values: LDSResults) -> tuple[LDSResults, torch.Tensor]:
        with torch.no_grad():
            values = self.filter(values)
            values = self.smooth(values)
            ll = self._observed_loglikelihood(values)
        return (values, ll)

    def m_step(self, values: KalmanResults, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            stats = self._calculate_sufficient_statistics(values)
        return self._solve_parameters(values, stats, **kwargs)

    def fit(
            self,
            X: list[torch.Tensor|np.ndarray],
            n_iter: int = 1000,
            emtol: float = 1e-3,
            **maximization_args
        ) -> LDSResults:
        X = self._initialize_observations(X)
        self._initialize_globals()
        values = self._initialize_values(X)

        for i in range(n_iter):
            values,ll = self.e_step(values)
            self.m_step(values, **maximization_args)

            values.loglike.append(ll)

            if not torch.isfinite(values.loglike[-1]):
                print(f"Log-likelihood is NaN or Inf, stopping EM at iter {i}")
                break

            if i > 0 and abs((values.loglike[-1] - values.loglike[-2]) / values.loglike[-2]) < emtol:
                print(f"Converged after {i} epochs, exiting")
                break

            if i % 20 == 0:
                print(f"Iteration {i}: {ll.item()}")

        if i == n_iter - 1:
            warnings.warn(f"Failed to converge after {i} epochs, exiting")

        values = self.e_step(values)[0]
        values.loglike_full = self._complete_loglikelihood(
            values,
            self._calculate_sufficient_statistics(values)
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

        return values