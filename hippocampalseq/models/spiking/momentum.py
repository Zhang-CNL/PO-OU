import numpy as np
import torch 

import hippocampalseq.utils as hseu
from hippocampalseq.models.linear_gaussian_system import LDSStatistics
from .spiking_base import *

class Momentum(SpikingBase):
    r"""State-space model that includes momentum as a parameters.
    We have a uniform prior and our observation covariance shifts over time.

    One-dimensional form of the model where we use the following approximation:
    $$
    \frac{d}{dt}\begin{pmatrix} v_t \\ z_t \end{pmatrix} = 
        \begin{pmatrix} -\lambda & 0 \\ 1 & 0\end{pmatrix}\begin{pmatrix} v_t \\ z_t \end{pmatrix}
        + \begin{pmatrix} \sigma_v & 0 \\ 0 & 0\end{pmatrix}\xi_t
    $$
        $v_t$ and $z_t$ are the velocity and position respectively, and each has an x and a y component.
    $$
    \begin{pmatrix} v_1 \\ z_1 \end{pmatrix} = 
        \begin{pmatrix} 0 & 0 \\ 0 & N_{xy}/2 \end{pmatrix} 
        + \begin{pmatrix} \sigma^2 / (2\lambda) & 0 \\ 0 & N_{xy}^2/12\end{pmatrix}\xi_1
    $$
    We approximated our uniform distribution prior for the position as a gaussian distribution
    and used a stationary OU process prior for the velocity.
    """
    def __init__(self,
            dt: float, 
            environment_size: list[tuple[int,...]],
            bin_size: int, 
            place_fields: hseu.NDArray, 
            spikemat_train: list[hseu.NDArray],
            spikemat_valid: list[hseu.NDArray]|None=None,
            initialization_method: str|dict[str, hseu.NDArray] = 'uniform',
        ):
        r"""Initialize the momentum SSM.
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (list[tuple[int,...]]): List of coordinates corresponding to the bounds of the environment.
            bin_size (int): Size of individual bins in cm.
        """
        if place_fields.shape[-1] == 1:
            nzdim = 1
        else:
            nzdim = 2
        super().__init__(
            2,
            slice(nzdim,None),
            dt, 
            environment_size,
            bin_size,
            place_fields,
            spikemat_train,
            spikemat_valid,
            initialization_method
        )

        # Random initialization of parameters
        # Scale of ln(10) meters
        self.decay     = self.random_initializer("decay", (1,), method=initialization_method)
        self.diffusion = self.random_initializer("diffusion", (1,), method=initialization_method)
        self.n_parameters = 2

    def name(self):
        return "Momentum"

    def __str__(self):
        return (self.name()
            + f"\n    Decay = {self.decay.exp().item():.2f}"
            + f"\n    Diffusion = {self.diffusion.exp().item():.2f}"
        )


    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        r"""Construct the transition matrix in the form

        $$\begin{pmatrix}-\lambda\Delta t + 1& 0 \\ \Delta t &1\end{pmatrix}$$

        Returns:
            torch.Tensor: Transition matrix
        """
        I  = torch.eye(self.latent_dim)
        Z  = torch.zeros(self.latent_dim, self.latent_dim)
        If = torch.eye(self.augmented_dim)

        M1     = -torch.exp(self.decay) * I
        top    = torch.cat((M1, Z), dim=1)
        bottom = torch.cat((I , Z), dim=1)
        A = torch.cat((top, bottom), dim=0) * self.dt + If
        return A

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        r"""Construct the transition covariance matrix in the form

        $$\begin{pmatrix}\sigma_v\sqrt{\Delta t} & 0 \\ 0 & 0\end{pmatrix}$$

        Returns:
            torch.Tensor: Transition covariance
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        sigma_m = torch.exp(self.diffusion)**2 * self.dt * I
        top     = torch.cat((sigma_m, Z), dim=1)
        bottom  = torch.cat((Z, Z), dim=1)
        Gamma   = torch.cat((top, bottom), dim=0)
        return Gamma

    def _construct_emission_matrix(self, mode: str = "train") -> torch.Tensor:
        return torch.hstack((
            torch.zeros(self.emission_dim, self.latent_dim),
            torch.eye(self.emission_dim), 
        ))


    def _construct_initial_mean(self, mode: str = "train") -> torch.Tensor:
        r"""Construct prior for momentum SSM.
        We want $P(z_1|z_0)$ to be a uniform distribution $U(K) = 1/K$, so we approximate this using
        a wide gaussian (large variance) since it approaches uniform.
        $$P(z_1) = U(N_x,N_y) \approx \mathcal{N}\left(
            \begin{bmatrix}N_x/2 \\ N_y/2\end{bmatrix},
            \begin{bmatrix}N_x^2/12 & 0 \\ 0 & N_y^2/12
            \end{bmatrix}\right)$$

        The prior for the velocity is a stationary OU process prior:
            $$P(v_1) = \mathcal{N}(v_1|0, \sigma^2 / (2\lambda))$$
        but if we take it in discrete time, we get a prior:
            $P(v_1) = \mathcal{N}\left(v_1|0, \sigma^2\Delta t / (1 - (1-\lambda\Delta t)^2)\right)$
        Returns:
            (torch.Tensor): Prior mean for augmented state $[z_t; z_{t-1}]^T$
            (torch.Tensor): Prior covariance for augmented state
        """
        diffs  = torch.tensor([es[1] + es[0] for es in self.environment_size])
        starts = torch.tensor([es[0] for es in self.environment_size])
        zmean  = (diffs / 2 + starts)[:,None]
        vmean  = torch.zeros((self.latent_dim, 1))
        return torch.cat((
            vmean, zmean
        ), dim=0)

    def _construct_initial_covariance(self, mode: str = "train") -> torch.Tensor:
        diffs = torch.tensor([es[1] + es[0] for es in self.environment_size])
        zcov = torch.diag(diffs)**2 / 12

        sigma = torch.exp(self.diffusion)
        lmb   = torch.exp(self.decay)
        vcov = torch.eye(self.latent_dim) * (sigma**2 * self.dt / (1 - (1 - lmb*self.dt)**2))
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        return torch.cat((
            torch.cat((vcov, Z), dim=1),
            torch.cat((Z, zcov), dim=1)
        ), dim=0)

    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        return torch.zeros((self.augmented_dim, 1))

    def _construct_emission_bias(self, mode: str = "train") -> torch.Tensor:
        return torch.zeros((self.emission_dim, 1))

    def _complete_loglikelihood(self, values: LDSResults, _: KalmanStatistics, mode: str = "train") -> torch.Tensor:
        r"""Calculate the full-data log-likelihood for the model given the current state.

        Args:
            values (LDSResults): The current decoded values for hidden states.
            mode (str): Either "train", "valid" or "test".
            _ (KalmanStatistics): Ignored for this model.

        Returns:
            torch.Tensor: The log-likelihood for the model.
        """
        loglike = 0

        log2pi = torch.log(2*torch.tensor(torch.pi))
        rank   = self.latent_dim

        self._initialize_globals(mode)


        for i in range(len(values.observations)):
            params = self.build_batch_parameters(i, mode=mode)
            T = values.observations[i].shape[0]
            t = torch.arange(T)
            F = hseu.extract_last_dims(params.transition_matrix, t[1:])
            Q = hseu.extract_last_dims(params.transition_covariance, t[1:])
            H = hseu.extract_last_dims(params.emission_matrix, t)
            R = hseu.extract_last_dims(params.emission_covariance, t)
            b = hseu.extract_last_dims(params.transition_bias, t[1:])
            d = hseu.extract_last_dims(params.emission_bias, t)
            mu0 = params.initial_mean
            V0  = params.initial_covariance

            if Q.ndim == 2:
                Q_v        = Q[:rank,:rank]
                logdet_Q_v = (T-1) * torch.logdet(Q_v)
            else:
                Q_v        = Q[:,:rank,:rank]
                logdet_Q_v = torch.sum(torch.logdet(Q_v), axis=0)

            muhat = values.smoothed_mean[i]
            vhat  = values.smoothed_cov[i]
            Jhat  = values.smoothed_gain[i]

            Exx  = values.observations[i] @ values.observations[i].mT
            Ezz  = vhat + muhat @ muhat.mT
            Ezx  = muhat @ values.observations[i].mT
            Ezz1 = vhat[1:] @ Jhat[:-1].mT + muhat[1:] @ muhat[:-1].mT 

            ip1 = Ezz[0]
            ip2 = mu0 @ muhat[0].mT
            ip3 = mu0 @ mu0.mT
            ill = hseu.mulinv(V0, ip1 - ip2 - ip2.mT + ip3)
            loglike += (
                torch.logdet(V0)
                + torch.trace(ill)
                + self.augmented_dim * log2pi
            )

            tp1 = Ezz[1:,:rank,:rank]
            tp2 = (Ezz1 @ F.mT)[:,:rank,:rank]
            tp3 = (F @ Ezz[:-1] @ F.mT)[:,:rank,:rank]
            tll = hseu.mulinv(Q_v, tp1 - tp2 - tp2.mT + tp3)
            tll = torch.sum(tll, axis=0)
            loglike += (
                logdet_Q_v
                + torch.trace(tll)
                + (T-1) * rank * log2pi
            )

            ep1 = Exx
            ep2 = H @ Ezx
            ep3 = H @ Ezz @ H.mT
            ell = hseu.mulinv(R, ep1 - ep2 - ep2.mT + ep3)
            ell = torch.sum(ell, axis=0)
            loglike += (
                torch.sum(torch.logdet(R), axis=0)
                + torch.trace(ell)
                + T * self.emission_dim * log2pi
            )

        return -0.5 * loglike

    def _calculate_sufficient_statistics(self, values: LDSResults, mode: str="train") -> KalmanStatistics:
        """
        Calculate sufficient statistics for performing maximization given the filtered
        and smoothed values of the model.
        Avoids calculating various unused values from the Kalman filtering version.

        Args:
            values (LDSResults): The filtered and smoothed values of the model.

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
            cov = btrace(v_vv[1:] @ g_vv[:-1].mT).reshape(-1) # tr(2,2) -> (1,1)
            # $$E[z_t] = \hat{\mu}_t$$
            ez = muvv # (2,1)
            # $$E[z_t^T z_t] = \hat{\mu}_t^T \hat{\mu}_t + tr(\hat{V}_t)$$
            ezz = (muvv.mT @ muvv).squeeze() + btrace(v_vv).reshape(-1) # (1,2)(2,1) + (1,1) -> (1,1)
            # $$E[z_t^T z_{t-1}] = tr(Cov[z_t,z_{t-1}]) + \hat{\mu}_t^T \hat{\mu}_{t-1}$$
            ezz1 = cov.squeeze() + (muvv[1:].mT @ muvv[:-1]).reshape(-1) # (1,1) + (1,2)(2,1) -> (1,1)

            Cov.append(cov)
            Ez.append(ez)
            Ezz.append(ezz)
            Ezz1.append(ezz1)

        return LDSStatistics(
            Cov=Cov,
            Ez=Ez,
            Ezz=Ezz,
            Ezz1=Ezz1,
            Ez1z=None,
            Exx=None,
            Exz=None,
            Ezx=None
        )

    def _solve_parameters(
            self, 
            values: LDSResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
        ) -> torch.Tensor:
        """Estimate all relevant parameters for the momentum SSM using autograd.

        Args:
            values (LDSResults): Momentum filtering pass results.
            stats (SufficientStatistics): Sufficient statistics from the Kalman filter/smoother.
            optimizer (str): The optimizer to use. Defaults to "Adam".
            lr (float): Learning rate for the optimizer. Defaults to 0.01.
            n_epochs (int): Number of epochs for SGD. Defaults to 1000.
            gd_tol (float): Tolerance for SGD. Defaults to 1e-03.

        Returns:
            torch.Tensor: The final negative log likelihood.
        """
        params = [
            hseu.grad_tensor(self.decay),
            hseu.grad_tensor(self.diffusion)
        ]

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

                loss = Ezz[i][1:] \
                    - 2 * M * Ezz1[i] \
                    + M**2 * Ezz[i][:-1]
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

        self._initialize_globals()

        return loss
