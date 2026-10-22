import torch
import hippocampalseq.utils as hseu
from .spiking_base import count_spikes, lag_spikes
from .ou_dynamics import OUDynamics


class OUDynamicsScaled(OUDynamics):
    r"""Version of the OUDynamics model where the 
    transition matrix, bias, and covariance are scaled by the spike count
    from the previous time step.
    $$\begin{aligned}
        z_{t+1} &= \tau_E^{-1}F z_t + \tau_E^{-1}b_t + \sqrt{\tau_E^{-1}}\xi_t \\
        \tau_E &= c\cdot n_{E,t} + 1
    \end{aligned}$$
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
            initialization_method: str|dict[str, hseu.NDArray] = 'hotstart'
        ):
        if initialization_method == 'hotstart':
            print('Hotstarting the model with no spike-scaling considered...')
            ou_model = OUDynamics(
                dt,
                environment_size,
                bin_size,
                place_fields,
                spikemat_train,
                true_position_train,
                spikemat_valid,
                true_position_valid,
            )
            results = ou_model.fit()
            initialization_method = {
                'F11'                     : ou_model.F11,
                'F12'                     : ou_model.F12,
                'F22'                     : ou_model.F22,
                'velocity_variance'       : ou_model.velocity_variance,
                'position_variance'       : ou_model.position_variance,
                'velocity_prior_variance' : ou_model.velocity_prior_variance
            }
            print('Hotstarting complete.')
        super().__init__(
            dt,
            environment_size,
            bin_size,
            place_fields,
            spikemat_train,
            true_position_train,
            spikemat_valid,
            true_position_valid
        ) 

        self.spikecount_train = [count_spikes(spk) for spk in spikemat_train]
        if spikemat_valid is not None:
            self.spikecount_valid = [count_spikes(spk) for spk in spikemat_valid]

        if isinstance(initialization_method, dict):
            self.initialization_method = 'uniform'

        self.tau = self.random_initializer(
            "tau",
            (1,),
            self.initialization_method
        )
        self.n_parameters += 1 

    def name(self):
        return "Spike-count Scaled OU Process Dynamics"

    def __str__(self):
        return (super().__str__()
            + "\n    "
            + rf"$\tau$ = {self.tau.exp().item():.2f}"
        )

    def build_denom(self, tau: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        n_t = lag_spikes(spikes)
        return hseu.atleast_3d(1.0 + tau * n_t)

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$I + \Delta t\begin{pmatrix}
            F_{11} & F_{12} \\
            -\tau_E^{-1}F_{12} & \tau_E^{-1}F_{22} \\
        \end{pmatrix}$$
        """
        if mode == "train":
            spikecount = self.spikecount_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
        elif mode == "test":
            spikecount = self.spikecount_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        I = torch.eye(self.latent_dim)

        F_scaled = []
        for spk in spikecount:
            T = len(spk)
            It = I.expand(T,-1,-1)
            denom = self.build_denom(self.tau.exp(), spk)
            Ftop = torch.cat((self.F11 * It, self.F12 * It), dim=2)
            Fbot = torch.cat((
                -self.F12 / denom * It,
                 self.F22 / denom * It
            ), dim=2)
            F = torch.cat((Ftop, Fbot), dim=1)
            F = torch.eye(self.augmented_dim) - F * self.dt
            F_scaled.append(F)
        return F_scaled

    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$-\Delta t\begin{pmatrix}
            F_{12} \\
            \tau_E^{-1}F_{22} \\
        \end{pmatrix}x_t$$
        """
        if mode == "train":
            spikecount = self.spikecount_train
            true_pos   = self.true_position_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
            true_pos   = self.true_position_valid
        elif mode == "test":
            spikecount = self.spikecount_test
            true_pos   = self.true_position_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        b_scaled = []
        for tp,spk in zip(true_pos,spikecount):
            denom = self.build_denom(self.tau.exp(), spk)
            b = torch.cat((
                self.F12 * self.dt * tp,
                self.F22 * self.dt * tp / denom
            ), dim=1)
            b_scaled.append(b)
        return b_scaled

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        r"""
        $$\begin{pmatrix}
            \sigma_v^2\Delta t & 0 \\ 0 & \sigma_z^2\Delta t \tau_E^{-1}
        \end{pmatrix}$$
        """
        if mode == "train":
            spikecount = self.spikecount_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
        elif mode == "test":
            spikecount = self.spikecount_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        Z    = torch.zeros((self.latent_dim, self.latent_dim))
        I    = torch.eye(self.latent_dim)
        sigv = torch.exp(self.velocity_variance)**2 * self.dt
        sigz = torch.exp(self.position_variance)**2 * self.dt

        Q_scaled = []
        for spk in spikecount:
            T = len(spk)
            denom = self.build_denom(self.tau.exp(), spk)
            _Z = Z.expand(T,-1,-1)
            _I = I.expand(T,-1,-1)
            _sigv = sigv * _I
            _sigz = sigz / denom * _I
            Q_s = torch.cat((
                torch.cat((_sigv, _Z), dim=2),
                torch.cat((_Z, _sigz), dim=2)
            ), dim=1)
            Q_scaled.append(Q_s)
        return Q_scaled


    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_matrix     = params.transition_matrix[batch]
        params.transition_covariance = params.transition_covariance[batch]
        # Transition bias already handled up above.
        return params

    def _solve_parameters(self, values: LDSResults, stats: LDSStatistics, optimizer: str = "Adam", lr: float = 0.01, n_epochs: int = 1000, gd_tol: float = 0.001) -> torch.Tensor:
        params = [
            hseu.grad_tensor(self.F11),
            hseu.grad_tensor(self.F12),
            hseu.grad_tensor(self.F22),
            hseu.grad_tensor(self.velocity_variance),
            hseu.grad_tensor(self.position_variance),
            hseu.grad_tensor(self.velocity_prior_variance),
            hseu.grad_tensor(self.tau),
        ]

        def loss_closure(params, n_batches: int, stats: SufficientStatistics):
            (
                F11,F12,F22,
                velocity_variance,
                position_variance,
                velocity_prior_variance,
                tau 
            ) = params

            Ez, Ezz, Ezz1 = stats.Ez, stats.Ezz, stats.Ezz1

            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))

            sigv = torch.exp(velocity_variance)**2 * self.dt
            sigz = torch.exp(position_variance)**2 * self.dt

            Ftop = torch.cat((F11 * I, F12 * I), dim=1)
            Qtop = torch.cat((sigv * I, Z), dim=1)

            tau = torch.exp(tau)
            sigv0 = torch.exp(velocity_prior_variance)**2 * self.dt
            true_position = self.true_position_train
            spikecount = self.spikecount_train
            total_loss = 0

            for i in range(n_batches):
                T = len(Ez[i])
                tpos = true_position[i][1:]
                denom = self.build_denom(tau, spikecount[i])[1:]
                _I = I.expand(T-1,-1,-1)
                Fscaled = torch.cat((
                    Ftop.expand(T-1,-1,-1),
                    torch.cat((-F12 * _I / denom, F22 * _I / denom), dim=2)
                ), dim=1)
                Fscaled = torch.eye(self.augmented_dim) - Fscaled * self.dt

                Qscaled = torch.cat((
                    Qtop.expand(T-1,-1,-1),
                    torch.cat((Z, sigz * I), dim=1) / denom
                ), dim=1)

                b_scaled = torch.cat((
                    F12 * tpos,
                    F22 * tpos / denom
                ), dim=1) * self.dt

                iloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                iloss = self.latent_dim * torch.log(sigv0) + iloss / sigv0

                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ Fscaled.mT
                tl3 = Fscaled @ Ezz[i][:-1] @ Fscaled.mT
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ b_scaled.mT
                bl2 = Fscaled @ Ez[i][:-1] @ b_scaled.mT
                bl3 = b_scaled @ b_scaled.mT
                bloss = bl3 + bl2 + bl2.mT - bl1 - bl1.mT

                tloss = hseu.mulinv(Qscaled, tloss + bloss)
                tloss = torch.sum(tloss, axis=0)
                tloss = (
                    torch.sum(torch.logdet(Qscaled), axis=0) 
                    + torch.trace(tloss)
                )

                total_loss += iloss + tloss
            
            return total_loss

        loss,params = hseu.optimize(
            loss_closure,
            params, 
            {
                'n_batches' : len(values.observations),
                'stats' : stats
            },
            {
                'optimizer' : optimizer,
                'lr'        : lr,
                'n_epochs'  : n_epochs,
                'gd_tol'    : gd_tol
            }
        )

        self.F11                     = params[0].detach()
        self.F12                     = params[1].detach()
        self.F22                     = params[2].detach()
        self.velocity_variance       = params[3].detach()
        self.position_variance       = params[4].detach()
        self.velocity_prior_variance = params[5].detach()
        self.tau                     = params[6].detach()

        return loss

    def transform(self,
            spikemats: list[hseu.NDArray],
            X: list[hseu.NDArray]
        ):

        self.spikecount_test = [count_spikes(spk) for spk in spikemats]
        return super().transform(spikemats, X)