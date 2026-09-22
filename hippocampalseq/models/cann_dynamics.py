import torch
from functools import wraps

from .momentum import *
from .linear_gaussian_system import *
import hippocampalseq.utils as hseu

__all__ = [
    'CANNDynamics',
    'bdiag'
]

bdiag = torch.vmap(torch.diag)

class CANNDynamics(Momentum):
    def __init__(self, 
            dt: float,
            environment_size: list[tuple[int,...]],
            bin_size: int,
            place_fields: hseu.NDArray,
            spikemat_train: hseu.NDArray,
            true_position_train: list[hseu.NDArray], 
            true_position_valid: list[hseu.NDArray]|None = None,
            spikemat_valid: hseu.NDArray|None = None,
            initialization_method: str|dict[str, hseu.NDArray] = 'uniform',
        ):
        r"""Initialize the CANNDynamics model.
        Model based on CANN subspace dynamics.

        $$\frac{d}{dt}\begin{pmatrix} v_t \\ z_t \end{pmatrix} = 
            \begin{pmatrix} -\lambda & 0 \\ 1 & 0\end{pmatrix}\begin{pmatrix} v_t \\ z_t \end{pmatrix} 
            + U\begin{pmatrix} 0 \\ (x_t-z_t) \end{pmatrix}
            + \begin{pmatrix} \sigma_v & 0 \\ 0 & \sigma_z\end{pmatrix}\xi_t
        $$

        Args:
            true_position (list[np.ndarray]): List of true positions.
        """
        super().__init__(
            dt,
            environment_size,
            bin_size,
            place_fields,
            spikemat_train,
            spikemat_valid,
            initialization_method
        )

        self.true_position_train = [
            hseu.atleast_3d(hseu.ensure_torch(tp))
            for tp in true_position_train
        ]
        if true_position_valid is not None:
            self.true_position_valid = [
                hseu.atleast_3d(hseu.ensure_torch(tp))
                for tp in true_position_valid
            ]

        self.approximate_covariance_diag = [
            bdiag(bdiag(cov)) 
            for cov in self.approximate_covariance
        ]
        if self.validation_approximate_covariance is not None:
            self.validation_approximate_covariance_diag = [
                bdiag(bdiag(cov))
                for cov in self.validation_approximate_covariance
            ]


        self.syn_input    = self.random_initializer(
            "synaptic_input",
            (1,),
            initialization_method
        ) # U
        self.pos_variance = self.random_initializer(
            "position_variance", 
            (1,), 
            initialization_method
        ) # sigma_z

        self.n_parameters  += 2

    def name(self):
        return "Momentum + Covariance Scaling & Synaptic Input"

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        r"""Construct the transition matrix:
        $$\begin{pmatrix}
            -\lambda \Delta t + 1 & 0 \\ \Delta t & -U \Delta t + 1
        \end{pmatrix}$$
        """
        if mode == "train":
            cov = self.approximate_covariance_diag
        elif mode == "valid":
            cov = self.validation_approximate_covariance_diag
        elif mode == "test":
            cov = self.testing_approximate_covariance_diag
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        If = torch.eye(self.augmented_dim)

        M1 = -torch.exp(self.decay) * I
        top = torch.cat((M1, Z), dim=1)
        #M4 = -torch.diag(torch.exp(self.syn_input))
        M4 = -torch.exp(self.syn_input)
        Fs = []
        for cc in cov:
            c = cc.clone()
            c[1:] = cc[:-1]
            _I = I.expand(len(c),-1,-1)
            _t = top.expand(len(c),-1,-1)
            bottom = torch.cat((_I, M4 * c), dim=2)
            F = torch.cat((_t, bottom), dim=1) * self.dt + If
            Fs.append(F)
        return Fs

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
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

    def _construct_transition_bias(self, mode: str = "train"):
        r"""Transition bias from synaptic input and true position.
        $$\begin{pmatrix}
            0 \\ U\Delta t x_t
        \end{pmatrix}$$
        """
        if mode == "train":
            tpos = self.true_position_train
            cov = self.approximate_covariance_diag
        elif mode == "valid": 
            tpos = self.true_position_valid
            cov = self.validation_approximate_covariance_diag
        elif mode == "test":
            tpos = self.true_position_test
            cov = self.testing_approximate_covariance_diag
        else:
            raise ValueError(f"Mode {mode} not recognized")
        b = []
        for tp,cc in zip(tpos, cov):
            c = cc.clone()
            c[1:] = cc[:-1]
            b.append(
                torch.cat((
                    torch.zeros_like(tp), 
                    (self.dt * torch.exp(self.syn_input) * c) @ tp
                ), dim=1)
            )
        return b

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_matrix = self.global_parameters.transition_matrix[batch]
        params.transition_bias = self.global_parameters.transition_bias[batch]
        return params

    def _calculate_sufficient_statistics(self, values: MomentumResults) -> KalmanStatistics:
        return LinearGaussianSystem._calculate_sufficient_statistics(self, values)

    def _complete_loglikelihood(self, values: MomentumResults, stats: KalmanStatistics, mode: str = "train") -> torch.Tensor:
        return LinearGaussianSystem._complete_loglikelihood(self, values, stats, mode)

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
                    torch.cat((self.dt * I, -U * self.dt * I + I), dim=1)
                ),
                dim=0
            )
            Ft = torch.cat((F1 * I, Z), dim=1)

            R = torch.cat(
                (
                    torch.cat((sigmav * I, Z), dim=1),
                    torch.cat((Z, sigmaz * I), dim=1)
                ),
                dim=0
            )

            v0 = sigmav / (1 - F1**2)

            total_loss = 0
            for i in range(n_batches):
                T = len(stats.Ez[i])

                true_position = self.true_position_train[i][1:]
                emission_cov = self.approximate_covariance_diag[i][:-1]

                Fb = torch.cat((
                    self.dt * I.expand(T-1,-1,-1), 
                    -self.dt * U * emission_cov + I), dim=2
                )
                F = torch.cat(
                    (
                        Ft.expand(T-1,-1,-1),
                        Fb
                    ),
                    dim=1
                )

                b = torch.cat((
                        torch.zeros_like(true_position),
                        (self.dt * U * emission_cov) @ true_position
                    ),
                    dim=1
                )

                # $2ln\ |V_0| + \mathbb{E}\left[z_1^T z_1\right] / V_0$
                ivloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                ivloss = self.latent_dim * torch.log(v0) + ivloss / v0

                # $\mathbb{E}\left[(z_1 - \mu_0 - b_1)^T \hat{V}_0^{-1} (z_1 - \mu_0 - b_1)\right]$
                # izloss = Ez[i][0,self.latent_dim:].mT @ ic @ (-b[0,self.latent_dim:]) \
                    # + im.mT @ ic @ b[0,self.latent_dim:] \
                    # - b[0,self.latent_dim:].mT @ ic @ Ez[i][0,self.latent_dim:] \
                    # + b[0,self.latent_dim:].mT @ ic @ im \
                    # + b[0,self.latent_dim:].mT @ ic @ b[0,self.latent_dim:]
                #izloss = Ez[i][0,self.latent_dim:].mT @ ic @ Ez[i][0,self.latent_dim:] \
                #    - Ez[i][0,self.latent_dim:].mT @ ic @ im \
                #    - im.mT @ ic @ Ez[i][0,self.latent_dim:] \
                #    + im.mT @ ic @ im

                
                iloss = ivloss #+ izloss

                # $ln\ |R| + \mathbb{E}\left[(z_t - Fz_{t-1} - b_t)^T R^{-1} (z_t - Fz_{t-1} - b_t)\right]$
                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ F.mT
                tl3 = F @ Ezz[i][:-1] @ F.mT 
                #tl1 = torch.sum(Ezz[i][1:], axis=0)
                #tl2 = torch.sum(Ezz1[i], axis=0) @ F.mT 
                #tl3 = F @ torch.sum(Ezz[i][:-1], axis=0) @ F.mT 
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ b.mT
                bl2 = F @ (Ez[i][:-1] @ b.mT)
                bl3 = b @ b.mT
                #bl1 = torch.sum(Ez[i][1:] @ b[1:].mT, axis=0)
                #bl2 = F @ torch.sum(Ez[i][:-1] @ b[1:].mT, axis=0)
                #bl3 = torch.sum(b[1:] @ b[1:].mT, axis=0)
                bloss = bl3 + bl2 + bl2.mT - bl1 - bl1.mT 

                loss = torch.sum(tloss + bloss, axis=0)
                loss = hseu.mulinv(R, loss)
                #loss = hseu.mulinv(R, tloss + bloss)
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

        print(f"Decay: {self.decay}")
        print(f"Diffusion: {self.diffusion}")
        print(f"Synaptic input: {self.syn_input}")
        print(f"Pos variance: {self.pos_variance}")

        self._initialize_globals("train")

        return loss

    def transform(self, spikemats: list[hseu.NDArray], Xtest: list[hseu.NDArray]):
        self.true_position_test = self._initialize_observations(Xtest)
        self.testing_emission_probability   = []
        self.testing_approximate_mean       = []
        self.testing_approximate_covariance = []

        for spk in spikemats:
            (
                emission_probability,
                approximate_mean,
                approximate_cov
            ) = self.spike_to_gaussian(spk)
            self.testing_emission_probability.append(emission_probability)
            self.testing_approximate_mean.append(approximate_mean)
            self.testing_approximate_covariance.append(approximate_cov)



        self.testing_approximate_covariance_diag = [
            bdiag(bdiag(cov))
            for cov in self.testing_approximate_covariance
        ]

        return LinearGaussianSystem.transform(
            self,
            self.testing_approximate_mean
        )