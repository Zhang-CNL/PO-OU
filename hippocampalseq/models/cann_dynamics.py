import torch
from functools import wraps

from .momentum import *
from .linear_gaussian_system import *
import hippocampalseq.utils as hseu

class CANNDynamics(Momentum):
    def __init__(self, true_position: list[np.ndarray], *args, **kwargs):
        r"""Initialize the CANNDynamics model.
        Model based on CANN subspace dynamics.

        $$\frac{d}{dt}\begin{pmatrix} v_t \\ z_t \end{pmatrix} = 
            \begin{pmatrix} -\lambda & 0 \\ 1 & 0\end{pmatrix}\begin{pmatrix} v_t \\ z_t \end{pmatrix} 
            + U\begin{pmatrix} 0 \\ (x_t-z_t) \end{pmatrix}
            + \begin{pmatrix} \sigma_v & 0 \\ 0 & \sigma_z\end{pmatrix}\xi_t
        $$

        Args:
            true_position (list[np.ndarray]): List of true positions.
            *args: Additional arguments for the parent class.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

        self.true_position = [
            hseu.atleast_3d(torch.from_numpy(tp))
            for tp in true_position
        ]

        self.syn_input    = torch.rand(1) # U
        self.pos_variance = torch.rand(1) # sigma_z

        self.n_parameters  += 2

    def _construct_transition_matrix(self) -> torch.Tensor:
        r"""Construct the transition matrix:
        $$\begin{pmatrix}
            -\lambda \Delta t + 1 & 0 \\ \Delta t & -U \Delta t + 1
        \end{pmatrix}$$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        If = torch.eye(self.augmented_dim)

        M1 = -torch.exp(self.decay) * I
        top = torch.cat((M1, Z), dim=1)
        M4 = -torch.exp(self.syn_input) * I
        bottom = torch.cat((I, M4), dim=1)
        F = torch.cat((top, bottom), dim=0) * self.dt + If
        return F

    def _construct_transition_covariance(self) -> torch.Tensor:
        r"""Transition covariance matrix.
        $$\begin{pmatrix}
            \sigma_v^2 \Delta t & 0 \\ 0 & \sigma_z^2 \Delta t
        \end{pmatrix}$$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        sigma_v = torch.exp(self.diffusion)**2 * self.dt * I
        sigma_z = torch.exp(self.pos_variance)**2 * self.dt * I
        top     = torch.cat((sigma_v, Z), dim=1)
        bottom  = torch.cat((Z, sigma_z), dim=1)
        Gamma   = torch.cat((top, bottom), dim=0)
        return Gamma

    def _construct_transition_bias(self):
        r"""Transition bias from synaptic input and true position.
        $$\begin{pmatrix}
            0 \\ U\Delta t x_t
        \end{pmatrix}$$
        """
        b = []
        for tp in self.true_position:
            b.append(
                torch.cat((
                    torch.zeros_like(tp), 
                    self.dt * torch.exp(self.syn_input) * tp
                ), dim=1)
            )
        return b

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        params = super().build_batch_parameters(batch)
        params.transition_bias = self.global_parameters.transition_bias[batch]
        return params

    def _calculate_sufficient_statistics(self, values: MomentumResults) -> KalmanStatistics:
        return LinearGaussianSystem._calculate_sufficient_statistics(self, values)

    def _complete_loglikelihood(self, values: MomentumResults, stats: KalmanStatistics) -> torch.Tensor:
        return LinearGaussianSystem._complete_loglikelihood(self, values, stats)

    def _solve_parameters(
            self, 
            values: MomentumResults, 
            stats: SufficientStatistics, 
            optimizer: str = "Adam", 
            lr: float = 0.01, 
            n_epochs: int = 1000, 
            gd_tol: float = 0.001
        ) -> torch.Tensor:
        decay        = hseu.grad_tensor(self.decay)
        diffusion    = hseu.grad_tensor(self.diffusion)
        syn_input    = hseu.grad_tensor(self.syn_input)
        pos_variance = hseu.grad_tensor(self.pos_variance)
        params = [decay, diffusion, syn_input, pos_variance]

        def loss_closure(
                params, 
                n_batches: int, 
                stats: SufficientStatistics
            ):
            decay,diffusion,syn_input,pos_variance = params

            Ez   = stats.Ez
            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1
            
            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))
            
            lmb   = torch.exp(decay)
            sigv  = torch.exp(diffusion)
            U     = torch.exp(syn_input)
            sigz  = torch.exp(pos_variance)

            F1 = -lmb * self.dt + 1
            sigmav = sigv**2 * self.dt
            sigmaz = sigz**2 * self.dt

            F = torch.cat(
                (
                    torch.cat((F1 * I, Z), dim=1),
                    torch.cat((self.dt * I, (-U * self.dt + 1) * I), dim=1)
                ),
                dim=0
            )
            R = torch.cat(
                (
                    torch.cat((sigmav * I, Z), dim=1),
                    torch.cat((Z, sigmaz * I), dim=1)
                ),
                dim=0
            )
            b = []
            for tp in self.true_position:
                b.append(
                    torch.cat(
                        (
                            torch.zeros_like(tp),
                            self.dt * U * tp
                        ),
                        dim=1
                    )
                )

            v0 = sigmav / (1 - F1**2)
            ic = torch.linalg.inv(
                self.global_parameters.initial_covariance[self.latent_dim:,self.latent_dim:]
            )
            im = self.global_parameters.initial_mean[self.latent_dim:]

            total_loss = 0
            for i in range(n_batches):
                T = len(stats.Ez[i])

                # $2ln\ |V_0| + \mathbb{E}\left[z_1^T z_1\right] / V_0$
                ivloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                ivloss = self.latent_dim * torch.log(v0) + ivloss / v0

                # $\mathbb{E}\left[(z_1 - \mu_0 - b_1)^T \hat{V}_0^{-1} (z_1 - \mu_0 - b_1)\right]$
                izloss = Ez[i][0,self.latent_dim:].mT @ ic @ (-b[i][0,self.latent_dim:]) \
                    + im.mT @ ic @ b[i][0,self.latent_dim:] \
                    - b[i][0,self.latent_dim:].mT @ ic @ Ez[i][0,self.latent_dim:] \
                    + b[i][0,self.latent_dim:].mT @ ic @ im \
                    + b[i][0,self.latent_dim:].mT @ ic @ b[i][0,self.latent_dim:]
                
                iloss = ivloss + izloss


                # $ln\ |R| + \mathbb{E}\left[(z_t - Fz_{t-1} - b_t)^T R^{-1} (z_t - Fz_{t-1} - b_t)\right]$
                tl1 = torch.sum(Ezz[i][1:], axis=0)
                tl2 = torch.sum(Ezz1[i], axis=0) @ F.mT 
                tl3 = F @ torch.sum(Ezz[i][:-1], axis=0) @ F.mT 
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = torch.sum(Ez[i][1:] @ b[i][1:].mT, axis=0)
                bl2 = F @ torch.sum(Ez[i][:-1] @ b[i][1:].mT, axis=0)
                bl3 = torch.sum(b[i][1:] @ b[i][1:].mT, axis=0)
                bloss = bl3 + bl2 + bl2.mT - bl1 - bl1.mT 

                loss = hseu.mulinv(R, tloss + bloss)
                loss = (T-1) * torch.logdet(R) + torch.trace(loss)

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
        
        self.decay        = params[0].detach()
        self.diffusion    = params[1].detach()
        self.syn_input    = params[2].detach()
        self.pos_variance = params[3].detach()

        self._initialize_globals()

        return loss