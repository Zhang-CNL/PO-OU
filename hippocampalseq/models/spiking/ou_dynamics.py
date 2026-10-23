import torch
import hippocampalseq.utils as hseu
from .spiking_base import *

# TODO: Try setting F_{12} in the bias term to 0.

class OUDynamics(SpikingBase):
    r"""Dynamics of a partially-observed OU process. Taken from eqn. 16 Chen 2025.
    $$\begin{pmatrix}v_{t+1} \\ z_{t+1} \end{pmatrix} = 
        \left(I - \Delta t\begin{pmatrix}F_{11} & F_{12} \\ -F_{12} & F_{22}\end{pmatrix}\right)
            \begin{pmatrix}v_t \\ z_t\end{pmatrix}
        + \Delta t \begin{pmatrix}F_{12} & 0 \\ 0 & F_{22}\end{pmatrix}
            \begin{pmatrix}x_t \\ x_t\end{pmatrix}
        + \begin{pmatrix}\sigma_v\sqrt{\Delta t} & 0 \\ 0 & \sigma_z\sqrt{\Delta t} \end{pmatrix}\xi_t
    $$
    """
    def __init__(self,
            dt: float,
            environment_size: list[tuple[int,...]],
            bin_size: int,
            place_fields: hseu.NDArray,
            spikemat_train: list[hseu.NDArray],
            true_position_train: list[hseu.NDArray],
            spikemat_valid: list[hseu.NDArray]|None=None,
            true_position_valid: list[hseu.NDArray]|None = None,
            initialization_method: str|dict[str, hseu.NDArray] = 'uniform'
        ):
        if place_fields.shape[-1] == 1:
            nzdim = 1
        else:
            nzdim = 2
        super().__init__(
            2,
            slice(nzdim, None),
            dt,
            environment_size,
            bin_size,
            place_fields,
            spikemat_train,
            spikemat_valid,
            initialization_method
        )

        self.true_position = {
            'train': self._initialize_observations(true_position_train)
        }
        if true_position_valid is not None:
            self.true_position['valid'] = self._initialize_observations(true_position_valid)

        self.F11 = self.random_initializer(
            "F11",
            (1,),
            method=initialization_method
        )
        self.F12 = self.random_initializer(
            "F12",
            (1,),
            method=initialization_method
        ) 
        self.F22 = self.random_initializer(
            "F22",
            (1,),
            method=initialization_method
        )
        self.velocity_variance = self.random_initializer(
            "velocity_variance",
            (1,),
            method=initialization_method
        )
        self.position_variance = self.random_initializer(
            "position_variance",
            (1,),
            method=initialization_method
        )
        self.velocity_prior_variance = self.random_initializer(
            "velocity_prior_variance",
            (1,),
            method=initialization_method
        )

        self.n_parameters = 6

    def name(self):
        return "OU Process Dynamics"

    def __str__(self):
        return (self.name()
            + f"\n    $F_{11}$ = {self.F11.item():.2f}"
            + f"\n    $F_{12}$ = {self.F12.item():.2f}"
            + f"\n    $F_{22}$ = {self.F22.item():.2f}"
            + f"\n    $\sigma_v$ = {self.velocity_variance.exp().item():.2f}"
            + f"\n    $\sigma_z$ = {self.position_variance.exp().item():.2f}"
            + f"\n    $\sigma_{{v,0}}$ = {self.velocity_prior_variance.exp().item():.2f}"
        )

    def _construct_initial_mean(self, mode: str = "train") -> torch.Tensor:
        diffs  = torch.tensor([es[1] + es[0] for es in self.environment_size])
        starts = torch.tensor([es[0] for es in self.environment_size])
        vmean  = torch.zeros((self.latent_dim, 1))
        zmean  = (diffs / 2 + starts)[:,None]
        return torch.cat((
            vmean, zmean
        ), dim=0)

    def _construct_initial_covariance(self, mode: str = "train") -> torch.Tensor:
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        I = torch.eye(self.latent_dim)
        diffs = torch.tensor([es[1] + es[0] for es in self.environment_size])
        zcov = torch.diag(diffs)**2 / 12

        sigv0 = torch.exp(self.velocity_prior_variance)**2 * self.dt
        vcov = sigv0 * I

        return torch.cat((
            torch.cat((vcov, Z), dim=1),
            torch.cat((Z, zcov), dim=1)
        ), dim=0)

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$I - \Delta t \begin{pmatrix} F_{11} & F_{12} \\
            -F_{12} & F_{22}\end{pmatrix}
        $$
        """
        I = torch.eye(self.latent_dim)
        F11 = self.F11 * I
        F12 = self.F12 * I
        F21 = F12
        F22 = self.F22 * I
        F = torch.vstack((
            torch.hstack(( F11, F12)), 
            torch.hstack((-F21, F22))
        ))
        return torch.eye(self.augmented_dim) - F * self.dt

    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$\Delta t \begin{pmatrix}F_{12} \\ F_{22}\end{pmatrix}x_t
        $$
        """
        true_pos = self.true_position[mode]
        bias = []
        for tp in true_pos:
            zs = self.F12 * self.dt * tp
            xs = self.F22 * self.dt * tp
            bias.append(
                torch.cat((zs, xs), dim=1)
            )
        return bias

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$\begin{pmatrix}\sigma_v^2\Delta t & 0 \\ 0 & \sigma_z^2\Delta t \end{pmatrix}
        $$
        """
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        I = torch.eye(self.latent_dim)
        sigv = torch.exp(self.velocity_variance)**2 * self.dt
        sigz = torch.exp(self.position_variance)**2 * self.dt
        return torch.vstack((
            torch.hstack((sigv * I, Z)), 
            torch.hstack((Z, sigz * I))
        ))

    def _construct_emission_matrix(self, mode: str = "train") -> torch.Tensor:
        return torch.hstack((
            torch.zeros(self.emission_dim, self.latent_dim),
            torch.eye(self.emission_dim), 
        ))

    def _construct_emission_bias(self, mode: str = "train") -> torch.Tensor:
        return torch.zeros((self.emission_dim, 1))

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_bias = self.global_parameters.transition_bias[batch]
        return params

    def _solve_parameters(
            self, 
            values: LDSResults,
            stats: SufficientStatistics,
            optimizer: str = "Adam",
            lr: float = .01, 
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3, 
        ) -> torch.Tensor:
        params = [
            hseu.grad_tensor(self.F11),
            hseu.grad_tensor(self.F12),
            hseu.grad_tensor(self.F22),
            hseu.grad_tensor(self.velocity_variance),
            hseu.grad_tensor(self.position_variance),
            hseu.grad_tensor(self.velocity_prior_variance),
        ]

        def loss_closure(params, n_batches: int, stats: SufficientStatistics):
            (
                F11,F12,F22,
                velocity_variance,
                position_variance,
                velocity_prior_variance
            ) = params

            Ez   = stats.Ez
            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1
            
            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))

            F = torch.cat((
                torch.cat(( F11 * I, F12 * I), dim=1),
                torch.cat((-F12 * I, F22 * I), dim=1)
            ), dim=0)
            F = torch.eye(self.augmented_dim) - F * self.dt

            sigv = torch.exp(velocity_variance)**2 * self.dt
            sigz = torch.exp(position_variance)**2 * self.dt

            Q = torch.cat((
                torch.cat((sigv * I, Z), dim=1),
                torch.cat((Z, sigz * I), dim=1)
            ), dim=0)

            sigv0 = torch.exp(velocity_prior_variance)**2 * self.dt# * I
            true_position = self.true_position['train']
            total_loss = 0
            for i in range(n_batches):
                T = len(Ezz[i])

                tpos = true_position[i][1:]
                bias = torch.cat((
                    F12 * tpos,
                    F22 * tpos
                ), dim=1) * self.dt

                iloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                iloss = self.latent_dim * torch.log(sigv0) + iloss / sigv0

                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ F.mT 
                tl3 = F @ Ezz[i][:-1] @ F.mT
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ bias.mT
                bl2 = F @ Ez[i][:-1] @ bias.mT
                bl3 = bias @ bias.mT
                bloss =  bl3 + bl2.mT - bl1 - bl1.mT

                tloss = torch.sum(tloss + bloss, axis=0)
                tloss = hseu.mulinv(Q, tloss)
                tloss = (T-1) * torch.logdet(Q) + torch.trace(tloss)

                total_loss += iloss + tloss

            return total_loss
        
        loss,params = hseu.optimize(
            loss_closure,
            params,
            {
                'n_batches' : len(values.observations),
                'stats'     : stats
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )
        
        self.F11 = params[0].detach()
        self.F12 = params[1].detach()
        self.F22 = params[2].detach()
        self.velocity_variance = params[3].detach()
        self.position_variance = params[4].detach()
        self.velocity_prior_variance = params[5].detach()

        self._initialize_globals()

        return loss

    def transform(self, spikemats: list[hseu.NDArray], X: list[hseu.NDArray]):
        self.true_position['test'] = self._initialize_observations(X)
        return super().transform(spikemats)