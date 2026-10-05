import torch
import hippocampalseq.utils as hseu
from .linear_gaussian_system import LDSStatistics
from .ou_dynamics import OUDynamics

def count_spikes(spikemat: torch.Tensor) -> torch.Tensor:
    spk = hseu.ensure_torch(spikemat).double()
    return hseu.atleast_3d(spk.sum(dim=1))

class OUDynamicsScaled(OUDynamics):
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
        super().__init__(
            dt=dt,
            environment_size=environment_size,
            bin_size=bin_size,
            place_fields=place_fields,
            spikemat_train=spikemat_train,
            true_position_train=true_position_train,
            spikemat_valid=spikemat_valid,
            true_position_valid=true_position_valid,
            initialization_method=initialization_method
        ) 

        self.spikecount_train = [count_spikes(spk) for spk in spikemat_train]
        if spikemat_valid is not None:
            self.spikecount_valid = [count_spikes(spk) for spk in spikemat_valid]

        self.tau = self.random_initializer(
            "tau",
            (1,),
            self.initialization_method
        )
        self.n_parameters += 1 

    def name(self):
        return "Spike-count Scaled OU Process Dynamics"

    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        if mode == "train":
            spikecount = self.spikecount_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
        elif mode == "test":
            spikecount = self.spikecount_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        F = super()._construct_transition_matrix(mode)
        tau = torch.exp(self.tau)

        F_scaled = []
        for spk in spikecount:
            T = len(spk)
            denom = 1.0 + tau * spk
            F_scaled.append(F.expand(T,-1,-1) / denom)
        return F_scaled

    def _construct_transition_bias(self, mode: str = "train") -> torch.Tensor:
        if mode == "train":
            spikecount = self.spikecount_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
        elif mode == "test":
            spikecount = self.spikecount_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        b = super()._construct_transition_bias(mode)
        tau = torch.exp(self.tau)

        b_scaled = []
        for bt,spk in zip(b,spikecount):
            denom = 1.0 + tau * spk
            b_scaled.append(bt / denom)
        return b_scaled

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        if mode == "train":
            spikecount = self.spikecount_train
        elif mode == "valid":
            spikecount = self.spikecount_valid
        elif mode == "test":
            spikecount = self.spikecount_test
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'valid', or 'test'.")

        Q = super()._construct_transition_covariance(mode)
        tau = torch.exp(self.tau)

        Q_scaled = []
        for spk in spikecount:
            T = len(spk)
            denom = 1.0 + tau * spk
            Q_scaled.append(Q.expand(T,-1,-1) / denom)
        return Q_scaled


    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        params = super().build_batch_parameters(batch, mode)
        params.transition_matrix     = params.transition_matrix[batch]
        params.transition_bias       = params.transition_bias[batch]
        params.transition_covariance = params.transition_covariance[batch]
        return params

    def _solve_parameters(self, values: MomentumResults, stats: LDSStatistics, optimizer: str = "Adam", lr: float = 0.01, n_epochs: int = 1000, gd_tol: float = 0.001) -> torch.Tensor:
        F11 = hseu.grad_tensor(self.F11)
        F12 = hseu.grad_tensor(self.F12)
        F22 = hseu.grad_tensor(self.F22)
        velocity_variance = hseu.grad_tensor(self.velocity_variance)
        position_variance = hseu.grad_tensor(self.position_variance)
        velocity_prior_variance = hseu.grad_tensor(self.velocity_prior_variance)
        tau = hseu.grad_tensor(self.tau)

        params = [
            F11, F12, F22, 
            velocity_variance, 
            position_variance, 
            velocity_prior_variance, 
            tau
        ]

        def loss_closure(params, n_batches: int, stats: SufficientStatistics):
            F11,F12,F22,velocity_variance,position_variance,velocity_prior_variance,tau = params

            Ez, Ezz, Ezz1 = stats.Ez, stats.Ezz, stats.Ezz1

            I = torch.eye(self.latent_dim)
            Z = torch.zeros((self.latent_dim, self.latent_dim))

            F = torch.cat((
                torch.cat(( F11 * I, -F12 * I), dim=1),
                torch.cat((-F12 * I, -F22 * I), dim=1)
            ), dim=0)
            F = (torch.eye(self.augmented_dim) - F * self.dt)

            velocity_variance = torch.exp(velocity_variance)
            position_variance = torch.exp(position_variance)
            sigv = velocity_variance**2 * self.dt
            sigz = position_variance**2 * self.dt

            Q = torch.cat((
                torch.cat(( sigz * I, Z), dim=1),
                torch.cat(( Z, sigv * I), dim=1)
            ), dim=0)

            sigv0 = torch.exp(velocity_prior_variance)**2 * self.dt
            true_position = self.true_position_train
            etau = torch.exp(tau)
            spikecount = self.spikecount_train
            total_loss = 0

            for i in range(n_batches):
                T = len(Ez[i])
                tpos = true_position[i][1:]
                denom = 1 + etau * spikecount[i][1:]
                Fscaled = F.expand(T-1, -1, -1) / denom

                Qscaled = Q.expand(T-1, -1, -1) / denom

                b_scaled = -torch.cat((
                    F12 * tpos,
                    F22 * tpos
                ), dim=1) * self.dt / denom

                iloss = Ez[i][0,:self.latent_dim].mT @ Ez[i][0,:self.latent_dim]
                iloss = self.latent_dim * torch.log(sigv0) + iloss / sigv0

                tl1 = Ez[i][1:]
                tl2 = Ezz1[i] @ Fscaled.mT
                tl3 = Fscaled @ Ezz[i][:-1] @ Fscaled.mT
                tloss = tl1 - tl2 - tl2.mT + tl3

                bl1 = Ez[i][1:] @ b_scaled.mT
                bl2 = Fscaled @ Ez[i][:-1] @ b_scaled.mT
                bl3 = b_scaled @ b_scaled.mT
                bloss = bl1 - bl2 - bl2.mT + bl3

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

        self.F11 = params[0].detach()
        self.F12 = params[1].detach()
        self.F22 = params[2].detach()
        self.velocity_variance = params[3].detach()
        self.position_variance = params[4].detach()
        self.velocity_prior_variance = params[5].detach()
        self.tau = params[6].detach()

        print(f"F_11 = {self.F11}")
        print(f"F_12 = {self.F12}")
        print(f"F_22 = {self.F22}")
        print(f"velocity_variance = {self.velocity_variance}")
        print(f"position_variance = {self.position_variance}")
        print(f"velocity_prior_variance = {self.velocity_prior_variance}")
        print(f"tau = {self.tau}")

        return loss

    def transform(self,
            spikemats: list[hseu.NDArray],
            Xtest: list[hseu.NDArray]|None = None
        ):

        self.spikecount_test = [count_spikes(spk) for spk in spikemats]
        return super().transform(spikemats, Xtest)