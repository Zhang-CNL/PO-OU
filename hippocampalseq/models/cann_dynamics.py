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

        self.syn_input    = torch.rand(1)
        self.pos_variance = torch.rand(1)

        self.augmented_dim += 2
        self.n_parameters  += 2

    def _construct_transition_matrix(self) -> torch.Tensor:
        r"""Construct the transition matrix:
        $$\begin{pmatrix}
            -\lambda \Delta t + 1 & 0 \\ \Delta t & -U \Delta t + 1
        \end{pmatrix}$$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        If = torch.eye(self.augmented_dim)

        M1 = -torch.exp(self.decay) * I
        top = torch.cat((M1, Z), dim=1)
        M4 = -torch.exp(self.syn_input) * I
        bottom = torch.cat((I, M4), dim=1)
        F = torch.cat((top, bottom), dim=0) * self.dt + If
        return F

    def _construct_transition_covariance(self) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))

        sigma_v = torch.exp(self.diffusion)**2 * self.dt * I
        sigma_z = torch.exp(self.pos_variance)**2 * self.dt * I
        top     = torch.cat((sigma_v, Z), dim=1)
        bottom  = torch.cat((Z, sigma_z), dim=1)
        Gamma   = torch.cat((top, bottom), dim=0)
        return Gamma

    def _construct_transition_bias(self):
        Z = torch.zeros((self.latent_dim, 1))
        b = []
        for tp in self.true_position:
            b.append(
                torch.cat((Z, self.dt * torch.exp(self.syn_input) * tp), dim=0)
            )
        return b

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        params = super().build_batch_parameters(batch)
        params.transition_bias = self.global_parameters.transition_bias[batch]
        return params

    def _calculate_sufficient_statistics(self, values: MomentumResults) -> KalmanStatistics:
        return LinearGaussianSystem._calculate_sufficient_statistics(self, values)

    def _solve_parameters(
            self, 
            values: MomentumResults, 
            stats: SufficientStatistics, 
            optimizer: str = "Adam", 
            lr: float = 0.01, 
            n_epochs: int = 1000, 
            gd_tol: float = 0.001
        ) -> torch.Tensor:

        decay        = torch.zeros(1, requires_grad=True)
        diffusion    = torch.zeros(1, requires_grad=True)
        syn_input    = torch.zeros(1, requires_grad=True)
        pos_variance = torch.zeros(1, requires_grad=True)
        with torch.no_grad():
            decay.copy_(self.decay)
            diffusion.copy_(self.diffusion)
            syn_input.copy_(self.syn_input)
            pos_variance.copy_(self.pos_variance)

        params = [decay, diffusion, syn_input, pos_variance]

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
        
        self.decay        = params[0].detach()
        self.diffusion    = params[1].detach()
        self.syn_input    = params[2].detach()
        self.pos_variance = params[3].detach()

        self._initialize_globals()

        return loss