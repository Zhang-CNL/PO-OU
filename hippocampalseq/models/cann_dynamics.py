import torch

from .momentum import *
import hippocampalseq.utils as hseu

class CANNDynamics(Momentum):
    def __init__(self, true_position: list[np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.true_position = [
            torch.from_numpy(tp) for tp in true_position
        ]
        

        self.syn_input = torch.rand(1)
        self.pos_variance = torch.rand(1)

        self.augmented_dim += 1
        self.n_parameters = 5

    def _construct_init_mean(self) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        init_mean = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_mean[:self.latent_dim, :self.latent_dim] = I
        init_mean[self.latent_dim:self.augmented_dim-1, :self.latent_dim] = I
        #init_mean[-1,-1] = 1.0
        return init_mean

    def _construct_init_var(self, initial_diffusion: torch.Tensor, jitter=0.0) -> torch.Tensor:
        I = torch.eye(self.latent_dim)
        init_cov = torch.zeros(self.augmented_dim, self.augmented_dim)
        init_cov[:self.latent_dim, :self.latent_dim] = initial_diffusion**2 * self.dt * I
        init_cov[self.latent_dim:self.augmented_dim-1, self.latent_dim:self.augmented_dim-1] = jitter * I
        return init_cov

    def _init_observation_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.obs_dim)
        Z = torch.zeros(self.obs_dim, self.augmented_dim-self.obs_dim)
        H = torch.cat((I, Z), dim=1)
        #H[-1,-1] = 1.0
        R = self.approximate_covariance
        return H, R

    def _construct_transition_mat(
            self, 
            decay: torch.Tensor, 
            syn_input: torch.Tensor,
            pos_variance: torch.Tensor,
        ) -> torch.Tensor:
        B = len(self.true_position)
        I = torch.eye(self.latent_dim)

        udt = syn_input * self.dt
        elt = torch.exp(-decay * self.dt)
        A1 = 1 + elt - udt
        A2 = udt - elt

        transition_matrices = []
        for b in range(B):
            T = len(self.true_position[b]) - 2

            B1 = syn_input * self.dt * self.true_position[b][2:][...,None]
            B2 = syn_input * self.dt * elt * self.true_position[b][1:-1][...,None]

            transitions = torch.zeros(T, self.augmented_dim, self.augmented_dim)
            transitions[:,:self.latent_dim,:self.latent_dim] = A1 * I
            transitions[:,:self.latent_dim,self.latent_dim:self.augmented_dim-1] = A2 * I

            transitions[:,:self.latent_dim,self.augmented_dim-1:] = B1 + B2
            transitions[:,self.latent_dim:self.augmented_dim-1,:self.latent_dim] = I
            transitions[:,-1,-1] = 1.0

            transition_matrices.append(transitions)

        return transition_matrices

    def _construct_transition_cov(
        self,
        decay: torch.Tensor,
        diffusion: torch.Tensor,    
        pos_variance: torch.Tensor
    ):
        I = torch.eye(self.latent_dim)
        Q = torch.zeros(self.augmented_dim, self.augmented_dim)
        q1 = (diffusion * self.dt)**2 / (2 * decay) \
            * (1 - torch.exp(-2 * decay * self.dt)) 
        q2 = pos_variance * self.dt
        q3 = -q2 * torch.exp(-decay * self.dt)
        Q[:self.latent_dim, :self.latent_dim] = (q1 + q2 + q3) * I
        return Q

    def _init_transition_matrices(self):
        A = self._construct_transition_mat(
            self.decay.exp(), 
            self.syn_input.exp(),
            self.pos_variance.exp(),
        )
        Q = self._construct_transition_cov(
            self.decay.exp(),
            self.diffusion.exp(),
            self.pos_variance.exp()
        )
        return A,Q

    def filter(self, values: MomentumResults) -> MomentumResults:
        transition_matrices = self.transition_matrices
        for batch in range(len(values.observations)):
            for t in range(1, len(values.observations[batch])):
                self.transition_matrices = transition_matrices[batch][t-2]
                values = self._filter(values, batch, t)
        self.transition_matrices = transition_matrices
        return values

    def smooth(self, values: MomentumResults) -> MomentumResults:
        transition_matrices = self.transition_matrices
        for batch in range(len(values.observations)):
            for t in reversed(range(len(values.observations[batch]) - 1)):
                self.transition_matrices = transition_matrices[batch][t-1]
                try: 
                    hseu.invmul(values.filtered_cov[batch][t] @ self.transition_matrices.T, values.predicted_cov[batch][t])
                except:
                    print(self.transition_matrices)
                    print(values.filtered_cov[batch][t])
                    print(values.predicted_cov[batch][t])
                    print(values.filtered_cov[batch][t] @ self.transition_matrices.T)
                    print(t)
                values = self._smooth(values, batch, t)
        self.transition_matrices = transition_matrices
        return values