import torch

import hippocampalseq.utils as hseu
from .cann_dynamics import *

class CANNDynamicsSpikes(CANNDynamics):
    def __init__(
            self,
            true_position: list[hseu.NDArray],
            place_fields: hseu.NDArray,
            spikemat: hseu.NDArray,
            *args,
            **kwargs
        ):
        r"""Projected CANN dynamics with a $\tau_E$ term added to the equation.

        $$\dot{z}_t = \tau_E^{-1} \left(v_t + U(x_t - z_t)\right) + \frac{\sigma_z}{\sqrt{\tau_E}}\xi_t$$

        In this system, $\tau_E = c\cdot n_{E,t}$ where $n_{E,t}$ is the number of excitatory spikes at a time point,
        and $c$ is a tunable parameter.
        """
        super().__init__(
            true_position, 
            place_fields=place_fields,
            spikemat=spikemat,
            *args, 
            **kwargs
        )

        self.n_spikes = [
            spk.sum(axis=1).reshape(-1,1,1) for spk in spikemat
        ]
        # Filter out all of the points with no spikes
        for i in range(len(self.n_spikes)):
            nonzero = (self.n_spikes[i] > 0).squeeze()
            self.n_spikes[i]               = self.n_spikes[i][nonzero]
            self.true_position[i]          = self.true_position[i][nonzero]
            self.emission_probabilities[i] = self.emission_probabilities[i][nonzero]
            self.approximate_mean[i]       = self.approximate_mean[i][nonzero]
            self.approximate_covariance[i] = self.approximate_covariance[i][nonzero]

        self.tau = torch.rand(1)
        self.n_parameters += 1

    def _construct_transition_matrix(self) -> list[torch.Tensor]:
        r"""Construct the transition matrix.
        $$\begin{pmatrix}
            -\lambda\Delta t + 1 & 0 \\
            \frac{\Delta t}{cn_{E,t}} & -\frac{U\Delta t}{cn_{E,t}} + 1
        \end{pmatrix}$$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        If = torch.eye(self.augmented_dim)

        M1 = -torch.exp(self.decay) * I
        M4 = -torch.exp(self.syn_input) * I
        top = torch.cat((M1, Z), dim=1)
        bottom = torch.cat((I, M4), dim=1)

        F = []
        for i in range(len(self.n_spikes)):
            T = len(self.n_spikes[i]) - 1
            _top = top.expand(T,-1,-1)
            _bottom = bottom.expand(T,-1,-1)
            _bottom = _bottom / (self.tau * self.n_spikes[i][1:])
            F.append(torch.cat((_top, _bottom), dim=1) * self.dt + If)
        return F

    def _construct_transition_covariance(self) -> list[torch.Tensor]:
        r"""Transition covariance matrices.
        $$\begin{pmatrix}
            \sigma_v^2 \Delta t & 0 \\ 0 & \frac{\sigma_z^2 \Delta t}{cn_{E,t}}
        \end{pmatrix}$$
        """
        I = torch.eye(self.latent_dim)
        Z = torch.zeros((self.latent_dim, self.latent_dim))
        sigmav = torch.exp(self.diffusion)**2 * self.dt * I
        sigmaz = torch.exp(self.pos_variance)**2 * self.dt * I

        gamma = []
        for i in range(len(self.n_spikes)):
            T = len(self.n_spikes[i]) - 1
            g1 = torch.cat((sigmav, Z), dim=1).expand(T,-1,-1)
            g2 = torch.cat((Z, sigmaz), dim=1).expand(T,-1,-1)
            g2 = g2 / (self.tau * self.n_spikes[i][1:])
            g= torch.cat((g1, g2), dim=1)
            gamma.append(g)

        return gamma

    def _construct_transition_bias(self) -> list[torch.Tensor]:
        r"""Transition bias
        $$\begin{pmatrix}
            0 \\ \frac{U\Delta t x_t}{cn_{E,t}}
        \end{pmatrix}$$
        """
        bias = super()._construct_transition_bias()
        for i in range(len(bias)):
            bias[i] /= self.tau * self.n_spikes[i]
        return bias

    def _construct_initial_mean(self) -> torch.Tensor:
        imean = super()._construct_initial_mean()
        initial_means = []
        for i in range(len(self.n_spikes)):
            _im = imean.clone()
            _im[self.latent_dim:] /= (self.tau * self.n_spikes[i][0])
            initial_means.append(_im)
        return torch.stack(initial_means)

    def _construct_initial_covariance(self) -> torch.Tensor:
        icov = super()._construct_initial_covariance()
        initial_covariances = []
        for i in range(len(self.n_spikes)):
            _ic = icov.clone()
            _ic[self.latent_dim:,self.latent_dim:] /= (self.tau * self.n_spikes[i][0])
            initial_covariances.append(_ic)
        return torch.stack(initial_covariances)

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        params = super().build_batch_parameters(batch)
        params.transition_matrix     = self.global_parameters.transition_matrix[batch]
        params.transition_covariance = self.global_parameters.transition_covariance[batch]
        params.initial_mean          = self.global_parameters.initial_mean[batch]
        params.initial_covariance    = self.global_parameters.initial_covariance[batch]
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
        decay = hseu.grad_tensor(self.decay)
        diffusion = hseu.grad_tensor(self.diffusion)
        syn_input = hseu.grad_tensor(self.syn_input)
        pos_variance = hseu.grad_tensor(self.pos_variance)
        tau = hseu.grad_tensor(self.tau)
        params = [decay, diffusion, syn_input, pos_variance, tau]


        self.decay = params[0].detach()
        self.diffusion = params[1].detach()
        self.syn_input = params[2].detach()
        self.pos_variance = params[3].detach()
        self.tau = params[4].detach()

        self._initialize_globals()

        return 0.0