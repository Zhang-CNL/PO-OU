import torch
from functools import wraps

from .momentum import *
from .kalman_filter import KalmanFilter
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
            torch.from_numpy(tp) for tp in true_position
        ]

        self.syn_input    = torch.rand(1)
        self.pos_variance = torch.rand(1)

        self.augmented_dim += 2
        self.n_parameters  += 2

    def _construct_init_mean(self) -> torch.Tensor:
        Z = torch.zeros(self.latent_dim, 1)
        initial_means = []
        for b in range(len(self.true_position)):
            imean = torch.cat(
                (Z, self.true_position[b][0][:,None], Z), dim=0
            )
            imean = hseu.atleast_2d(imean)
            initial_means.append(imean)
        return initial_means

    def _construct_init_var(self, 
            decay: torch.Tensor, 
            diffusion: torch.Tensor, 
            syn_input: torch.Tensor,
            pos_variance: torch.Tensor
        ) -> torch.Tensor:
        """Construct the initial variance of the augmented state matrix.
        We model both the velocity and position as stationary OU processes.
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        vc = diffusion**2 / (2*decay) * I
        vz = pos_variance**2 / (2*syn_input) * I

        init_cov = torch.cat(
            (
                torch.cat((vc, Z, Z), dim=1),
                torch.cat((Z, vz, Z), dim=1),
                torch.cat((Z, Z, Z), dim=1),
            ),
            dim=0
        )

        return init_cov

    def _init_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        init_mean = self._construct_init_mean()
        init_cov  = self._construct_init_var(
            torch.exp(self.decay),
            torch.exp(self.diffusion),
            torch.exp(self.syn_input),
            torch.exp(self.pos_variance)
        )
        return init_mean, init_cov

    @wraps(Momentum._init_observation_matrices)
    def _init_observation_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)
        H = torch.cat((Z, I, Z), dim=1)
        R = self.approximate_covariance
        return H, R

    def _construct_transition_mat(
            self, 
            decay: torch.Tensor, 
            syn_input: torch.Tensor,
        ) -> torch.Tensor:
        r"""Construct transition matrices for the CANN dynamics model.
        $$\begin{pmatrix}
            -\lambda\Delta t + 1 & 0 & 0\\ 
            \Delta t & (1 - \Delta t U) & \Delta t U x_t \\
            0 & 0 & 1
            \end{pmatrix}
        $$
        """
        B = len(self.true_position)
        I = torch.eye(self.latent_dim)

        udt = syn_input * self.dt * I
        A1 = (1 - decay * self.dt) * I
        Idt = self.dt * I
        A2 = I - udt

        diag = torch.vmap(torch.diag)

        transition_matrices = []
        for b in range(B):
            x = self.true_position[b][1:]
            T = len(x)
            # TODO: Figure out how to vectorize torch.cat usage so that we can preserve gradient?
            A = torch.zeros(T, self.augmented_dim, self.augmented_dim)

            D1 = self.latent_dim 
            D2 = self.augmented_dim - self.latent_dim
            D3 = self.augmented_dim

            A[:,:D1,:D1]     = A1
            A[:,D1:D2,:D1]   = Idt
            A[:,D1:D2,D1:D2] = A2
            A[:,D1:D2,D2:D3] = diag((udt @ x).squeeze()) 
            A[:,D2:D3,D2:D3] = I

            transition_matrices.append(transitions)

        return transition_matrices

    def _construct_transition_cov(
            self,
            diffusion: torch.Tensor,    
            pos_variance: torch.Tensor
        ) -> torch.Tensor:
        r"""Construct transition covariance matrices for the CANN dynamics model.
        $$\begin{pmatrix}
            \sigma_v^2\Delta t & 0 & 0\\
            0 & \sigma_z^2\Delta t & 0\\
            0 & 0 & 0
            \end{pmatrix}
        $$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros(self.latent_dim, self.latent_dim)

        Q = torch.cat(
            (
                torch.cat((I * diffusion**2 * self.dt, Z, Z), dim=1),
                torch.cat((Z, I * pos_variance**2 * self.dt, Z), dim=1),
                torch.cat((Z, Z, Z), dim=1),
            ),
            dim=0
        )

        return Q

    @wraps(Momentum._init_transition_matrices)
    def _init_transition_matrices(self):
        A = self._construct_transition_mat(
            torch.exp(self.decay), 
            torch.exp(self.syn_input),
        )
        Q = self._construct_transition_cov(
            torch.exp(self.diffusion),
            torch.exp(self.pos_variance)
        )
        return A,Q

    @wraps(KalmanFilter.filter)
    def filter(self, values: MomentumResults) -> MomentumResults:
        transition_matrices = self.transition_matrices
        emission_covariance = self.observation_covariance
        initial_mean = self.initial_state_mean
        for batch in range(len(values.observations)):
            self.observation_covariance = emission_covariance[batch][0]
            self.initial_state_mean  = initial_mean[batch]
            values = self._filter_init(values, batch)

            for t in range(1, len(values.observations[batch])):
                self.transition_matrices = transition_matrices[batch][t-1]
                self.observation_covariance = emission_covariance[batch][t]
                values = self._filter(values, batch, t)

        self.initial_state_mean  = initial_mean
        self.transition_matrices = transition_matrices
        self.observation_covariance = emission_covariance
        return values

    @wraps(KalmanFilter.smooth)
    def smooth(self, values: MomentumResults) -> MomentumResults:
        transition_matrices = self.transition_matrices
        for batch in range(len(values.observations)):
            values = self._smooth_init(values, batch)
            for t in reversed(range(len(values.observations[batch]) - 1)):
                self.transition_matrices = transition_matrices[batch][t]
                values = self._smooth(values, batch, t)

        self.transition_matrices    = transition_matrices
        return values