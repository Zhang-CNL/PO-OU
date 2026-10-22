import torch

import hippocampalseq.utils as hseu
from .cann_dynamics import *
from .spiking_base import count_spikes,lag_spikes

# TODO: Rename this class

class CANNDynamicsSpikes(CANNDynamics):
    def __init__(
            self,
            *args,
            **kwargs
        ):
        r"""Projected CANN dynamics with a $\tau_E$ term added to the equation.

        $$\dot{z}_t = \tau_E^{-1} \left(z_t + U(x_t - z_t)\right) + \sigma_z\tau_E^{-1/2}\xi_t$$

        In this system, $\tau_E = c\cdot n_{E,t} + 1$ where $n_{E,t}$ is the number of excitatory spikes at a time point,
        and $c$ is a tunable parameter.
        """
        super().__init__(
            *args,
            **kwargs
        )

        spikemat_train = kwargs.get('spikemat_train')
        spikemat_valid = kwargs.get('spikemat_valid', None)
        self.n_spikes_train = [
            count_spikes(spk) for spk in spikemat_train
        ]
        if spikemat_valid is not None:
            self.n_spikes_valid = [
                count_spikes(spk) for spk in spikemat_valid
            ]

        self.tau = self.random_initializer(
            "tau",
            (1,),
            self.initialization_method
        )
        self.n_parameters += 1

    def name(self):
        return "Momentum + Synapic Input & Spike Scaling"

    def __str__(self):
        return (
            super().__str__()
            + f"\n    Spike Scaling = {self.tau.exp().item():.2f}"
        )

    def build_denom(self, tau: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        n_t = lag_spikes(spikes)
        return hseu.atleast_3d(1.0 + tau * n_t)

    def _construct_transition_matrix(self, mode: str = "train") -> list[torch.Tensor]:
        r"""Construct the transition matrix.
        $$\begin{pmatrix}
            1-\lambda\Delta t & 0 \\
            \frac{\Delta t}{cn_{E,t}} & 1-\frac{U\Delta t}{cn_{E,t}}
        \end{pmatrix}$$
        """
        if mode == "train":
            nspikes = self.n_spikes_train
        elif mode == "valid":
            nspikes = self.n_spikes_valid
        elif mode == "test":
            nspikes = self.n_spikes_test
        else:
            raise ValueError(f"Mode {mode} not recognized")
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        If = torch.eye(self.augmented_dim)

        M1 = -torch.exp(self.decay) * I
        M4 = -torch.exp(self.syn_input) * I
        top = torch.cat((M1, Z), dim=1)

        Fs = []
        for spk in nspikes:
            tspk = self.build_denom(self.tau.exp(), spk)
            Ftop = top.expand(len(tspk),-1,-1)
            Fbot = torch.cat((
                I.expand(len(tspk),-1,-1) / tspk,
                M4 / tspk
            ), dim=2)
            F = torch.cat((Ftop, Fbot), dim=1) * self.dt + If
            Fs.append(F)
        return Fs

    def _construct_transition_covariance(self, mode: str = "train") -> list[torch.Tensor]:
        r"""Transition covariance matrices.
        $$\begin{pmatrix}
            \sigma_v^2 \Delta t & 0 \\ 0 & \frac{\sigma_z^2 \Delta t}{cn_{E,t}}
        \end{pmatrix}$$
        """
        if mode == "train":
            nspikes = self.n_spikes_train
        elif mode == "valid":
            nspikes = self.n_spikes_valid
        elif mode == "test":
            nspikes = self.n_spikes_test
        else:
            raise ValueError(f"Mode {mode} not recognized")
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        sigmav = torch.exp(self.diffusion)**2 * self.dt * I
        sigmaz = torch.exp(self.pos_variance)**2 * self.dt * I
        top = torch.cat((sigmav, Z), dim=1)

        Qs = []
        for spk in nspikes:
            tspk = self.build_denom(self.tau.exp(), spk)
            Qtop = top.expand(len(tspk),-1,-1)
            Qbot = torch.cat((
                Z.expand(len(tspk),-1,-1),
                sigmaz / tspk
            ), dim=2)
            Q = torch.cat((Qtop, Qbot), dim=1)
            Qs.append(Q)
        return Qs

    def _construct_transition_bias(self, mode: str = "train") -> list[torch.Tensor]:
        r"""Transition bias
        $$\begin{pmatrix}
            0 \\ \frac{U\Delta t x_t}{cn_{E,t}}
        \end{pmatrix}$$
        """
        if mode == "train":
            nspikes = self.n_spikes_train
            true_position = self.true_position_train
        elif mode == "valid":
            nspikes = self.n_spikes_valid
            true_position = self.true_position_valid
        elif mode == "test":
            nspikes = self.n_spikes_test
            true_position = self.true_position_test
        else:
            raise ValueError(f"Mode {mode} not recognized")

        I = torch.eye(self.latent_dim)
        Udt = torch.exp(self.syn_input) * self.dt * I

        biases = []
        for spk,tp in zip(nspikes, true_position):
            tspk = self.build_denom(self.tau.exp(), spk)
            top = torch.zeros_like(tp)
            bottom = Udt @ tp / tspk
            bias = torch.cat((top, bottom), dim=1)
            biases.append(bias)
        return biases

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_matrix     = self.global_parameters.transition_matrix[batch]
        params.transition_covariance = self.global_parameters.transition_covariance[batch]
        return params

    def _solve_parameters(
        self, 
        values: LDSResults, 
        stats: LDSStatistics, 
        optimizer: str = "Adam", 
        lr: float = 0.01, 
        n_epochs: int = 1000, 
        gd_tol: float = 0.001
    ) -> torch.Tensor:
        params = [
            hseu.grad_tensor(self.decay),
            hseu.grad_tensor(self.diffusion),
            hseu.grad_tensor(self.syn_input),
            hseu.grad_tensor(self.pos_variance),
            hseu.grad_tensor(self.tau),
        ]

        def loss_closure(
            params, 
            n_batches: int, 
            stats: LDSStatistics
        ):
            (
                decay, diffusion, 
                syn_input, pos_variance,
                tau
            ) = params

            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))
            If = torch.eye(self.augmented_dim)

            Ez = stats.Ez
            Ezz = stats.Ezz
            Ezz1 = stats.Ezz1

            lmb  = torch.exp(decay)
            sigv = torch.exp(diffusion)
            tau  = torch.exp(tau)
            U    = torch.exp(syn_input)
            sigz = torch.exp(pos_variance)

            spikes = self.n_spikes_train

            sigmav = sigv**2 * self.dt
            sigmaz = sigz**2 * self.dt * I

            Ftop = torch.cat((lmb * I, Z), dim=1)
            Qtop = torch.cat((sigmav * I, Z), dim=1)

            v0 = sigmav / (1 - (1 + lmb * self.dt)**2)

            total_loss = 0
            for i in range(n_batches):
                T     = len(stats.Ez[i])
                spike = self.build_denom(tau, spikes[i])[1:]
                tp    = self.true_position_train[i][1:]

                Fb = torch.cat((
                    I.expand(T-1,-1,-1) / spike,
                    -(U * I) / spike
                ), dim=2) 
                F = torch.cat((
                    Ftop.expand(T-1,-1,-1),
                    Fb
                ), dim=1) * self.dt + If

                Qb = torch.cat((
                    Z.expand(T-1,-1,-1),
                    sigmaz / spike
                ), dim=2)
                Q = torch.cat((
                    Qtop.expand(T-1,-1,-1),
                    Qb
                ), dim=1)

                b = torch.cat((
                    torch.zeros_like(tp),
                    U * self.dt * tp / spike
                ), dim=1)

                ivloss = self.latent_dim * torch.log(v0) \
                    + (Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]) / v0

                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ F.mT 
                tl3 = F @ Ezz[i][:-1] @ F.mT 
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ b.mT 
                bl2 = F @ Ez[i][:-1] @ b.mT
                bl3 = b @ b.mT 
                bloss = bl3 + bl2 + bl2.mT - bl1 - bl1.mT
                
                tloss = hseu.mulinv(Q, tloss + bloss)
                tloss = torch.sum(tloss, axis=0)
                tloss = (
                    torch.sum(torch.logdet(Q), axis=0)
                    + torch.trace(tloss)
                )

                total_loss += tloss + ivloss

            return total_loss

        loss,params = hseu.optimize(
            loss_closure,
            params,
            {
                'n_batches' : len(self.n_spikes_train),
                'stats'     : stats
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
        self.tau          = params[4].detach()

        self._initialize_globals("train")
        return loss

    def transform(self, spikemats: list[hseu.NDArray], Xtest: list[hseu.NDArray]):
        self.n_spikes_test = []
        for spk in spikemats:
            spk = spk.sum(axis=1)
            spk = hseu.atleast_3d(spk)
            self.n_spikes_test.append(torch.from_numpy(spk))

        return super().transform(spikemats, Xtest)
