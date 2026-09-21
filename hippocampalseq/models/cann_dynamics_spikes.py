from hippocampalseq.models.linear_gaussian_system import LDSResults
import torch
from scipy.ndimage import gaussian_filter1d

import hippocampalseq.utils as hseu
from .cann_dynamics import *

def _regularize_psd(self, cov: torch.Tensor, min_eig: float = 1e-8) -> None:
    """In-place regularization to enforce PSD via eigenvalue clamping."""
    U, S, Vh = torch.linalg.svd(cov)
    S_clamped = torch.clamp(S, min=min_eig)
    cov_reg = U @ torch.diag(S_clamped) @ Vh
    cov.copy_(cov_reg)

class CANNDynamicsSpikes(CANNDynamics):
    def __init__(
            self,
            place_fields: hseu.NDArray,
            spikemat_train: hseu.NDArray,
            spikemat_valid: hseu.NDArray,
            true_position_train: list[hseu.NDArray],
            true_position_valid: list[hseu.NDArray]|None = None,
            gaussian_sigma_s: float = 0.1,
            *args,
            **kwargs
        ):
        r"""Projected CANN dynamics with a $\tau_E$ term added to the equation.

        $$\dot{z}_t = \tau_E^{-1} \left(v_t + U(x_t - z_t)\right) + \frac{\sigma_z}{\sqrt{\tau_E}}\xi_t$$

        In this system, $\tau_E = c\cdot n_{E,t}$ where $n_{E,t}$ is the number of excitatory spikes at a time point,
        and $c$ is a tunable parameter.
        """
        super().__init__(
            true_position_train=true_position_train, 
            true_position_valid=true_position_valid,
            place_fields=place_fields,
            spikemat_train=spikemat_train,
            spikemat_valid=spikemat_valid,
            *args, 
            **kwargs
        )

        sigma = (gaussian_sigma_s / self.dt).item()

        self.n_spikes_train = []
        for spk in spikemat_train:
            #spk = gaussian_filter1d(spk, sigma=sigma, axis=0)
            spk = torch.from_numpy(spk).double()
            spk = hseu.atleast_3d(spk.sum(axis=1))
            #spk = spk / (spk + 1)
            self.n_spikes_train.append(spk)

        self.n_spikes_valid = []
        for spk in spikemat_valid:
            #spk = gaussian_filter1d(spk, sigma=sigma, axis=0)
            spk = torch.from_numpy(spk).double()
            spk = hseu.atleast_3d(spk.sum(axis=1))
            #spk = spk / (spk + 1)
            self.n_spikes_valid.append(spk)

        self.tau = torch.rand(1)
        self.n_parameters += 1

    def name(self):
        return "Momentum + Synapic Input & Spike Scaling"

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
        #M4 = -torch.exp(self.syn_input) * I
        M4 = torch.diag(-torch.exp(self.syn_input))
        top = torch.cat((M1, Z), dim=1)
        bottom = torch.cat((I, M4), dim=1)

        Fs = []
        for spk in nspikes:
            _spk = spk.clone()
            _spk[1:] = spk[:-1]
            tspk = self.tau * _spk
            _I = I.expand(len(_spk),-1,-1) / tspk
            _t = top.expand(len(_spk),-1,-1)
            bottom = torch.cat((_I, M4 / tspk), dim=2)
            F = torch.cat((_t, bottom), dim=1) * self.dt + If
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
            _spk = spk.clone()
            _spk[1:] = spk[:-1]
            _t = top.expand(len(_spk),-1,-1)
            _z = Z.expand(len(_spk),-1,-1)
            bottom = torch.cat((_z, sigmaz / (self.tau * _spk)), dim=2)
            Q = torch.cat((_t, bottom), dim=1)
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
        #Udt = torch.exp(self.syn_input) * self.dt * I
        Udt = torch.diag(torch.exp(self.syn_input)) * self.dt

        biases = []
        for spk,tp in zip(nspikes, true_position):
            _spk = spk.clone()
            _spk[1:] = spk[:-1]
            top = torch.zeros_like(tp)
            bottom = Udt @ tp / (self.tau * _spk)
            bias = torch.cat((top, bottom), dim=1)
            biases.append(bias)
        return biases

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_matrix     = self.global_parameters.transition_matrix[batch]
        params.transition_covariance = self.global_parameters.transition_covariance[batch]
        params.transition_bias       = self.global_parameters.transition_bias[batch]
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
        decay        = hseu.grad_tensor(self.decay)
        diffusion    = hseu.grad_tensor(self.diffusion)
        syn_input    = hseu.grad_tensor(self.syn_input)
        pos_variance = hseu.grad_tensor(self.pos_variance)
        tau          = hseu.grad_tensor(self.tau)
        params = [decay, diffusion, syn_input, pos_variance, tau]


        def loss_closure(
            params, 
            n_batches: int, 
            stats: LDSStatistics
        ):
            decay,diffusion,syn_input,pos_variance,tau = params

            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))
            If = torch.eye(self.augmented_dim)

            Ez = stats.Ez
            Ezz = stats.Ezz
            Ezz1 = stats.Ezz1

            lmb  = torch.exp(decay)
            sigv = torch.exp(diffusion)
            U    = torch.diag(torch.exp(syn_input))
            sigz = torch.exp(pos_variance)
            tau  = torch.exp(tau)

            spikes = self.n_spikes_train

            F1     = -lmb * self.dt + 1
            sigmav = sigv**2 * self.dt
            sigmaz = sigz**2 * self.dt
            udt    = U * self.dt# * I

            Ft = torch.cat((F1 * I, Z), dim=1)
            Qt = torch.cat((sigmav * I, Z), dim=1)

            v0 = sigmav / (1 - F1**2)

            total_loss = 0
            for i in range(n_batches):
                T     = len(stats.Ez[i])
                spike = spikes[i][:-1] * tau
                tp    = self.true_position_train[i][1:]
                udts  = udt / spike
                loss       = 0

                Fb = torch.cat((
                    self.dt * I.expand(T-1,-1,-1),
                    -udts
                ), dim=2)
                F = torch.cat((
                    Ft.expand(T-1,-1,-1),
                    Fb
                ), dim=1) + If

                Qb = torch.cat((
                    Z.expand(T-1,-1,-1),
                    sigmaz * I / spike
                ), dim=2)
                Q = torch.cat((
                    Qt.expand(T-1,-1,-1),
                    Qb
                ), dim=1)

                b = torch.cat((
                    torch.zeros_like(tp),
                    udts @ tp
                ), dim=1)

                ivloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                ivloss = self.latent_dim * torch.log(v0) + ivloss / v0
                loss = loss + ivloss

                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ F.mT 
                tl3 = F @ Ezz[i][:-1] @ F.mT 
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ b.mT 
                bl2 = F @ (Ez[i][:-1] @ b.mT)
                bl3 = b @ b.mT 
                bloss = bl1 - bl2 - bl2.mT + bl3
                
                tloss = hseu.mulinv(Q, tloss + bloss)
                tloss = torch.sum(tloss, axis=0)
                tloss = torch.sum(torch.logdet(Q), axis=0) + torch.trace(tloss)
                loss = loss + tloss

                total_loss += loss

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
