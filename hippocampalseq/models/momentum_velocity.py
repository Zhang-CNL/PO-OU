import torch

import hippocampalseq.utils as hseu
from .linear_gaussian_system import *
from .momentum import *

class MomentumVelocity(Momentum):
    def __init__(
            self, 
            place_fields: hseu.NDArray, 
            spikemat: list[hseu.NDArray],
            dt: float, 
            environment_size: list[tuple[int,...]],
            bin_size: int, 
            velocity_type: str ='true',
            seed: int|None = 42
        ):
        r"""Momentum model but we include an observed velocity variable.
        The same basic internal momentum is used, however the emission probabilities are different.
        The emission function is the linear expression:
        $$\begin{pmatrix} v_{true,t} \\ z_{true,t} \end{pmatrix} = 
            \begin{bmatrix} I_2 & 0_2 \\ 0_2 & I_2\end{bmatrix}
            \begin{pmatrix} v_t \\ z_t \end{pmatrix} + 
            w_t
        $$
        where 
        $$w_t \sim \mathcal{N}\left(0, \begin{pmatrix}
            K & 0 \\ 0 & \Sigma_t
        \end{pmatrix}\right)$$
        In the noise distribution, $\Sigma_t$ is from approximating the emission probabilities as a gaussian,
        and $K$ is a 2x2 matrix modeling the emission noise for the velocity variable.
        Velocity is a (T,1,1) variable in this case.

        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            spikemat (np.ndarray|torch.Tensor): (T, Ncells) Spikemat,
            dt (float): Time step for the transition matrix.
            environment_size (list[tuple[int,...]]): List of coordinates corresponding to the bounds of the environment.
            bin_size (int): Size of individual bins in cm.
            velocity_type (str): Type of velocity to use. 
                'observed' to calculate it from the place-fields. 'true' to use the true velocity.
                Defaults to 'true'.
            seed: (int|None): Seed for the random number generator
        """
        super().__init__(
            place_fields, 
            spikemat, 
            dt, 
            environment_size, 
            bin_size, 
            seed
        )
        self.velocity_type = velocity_type
        self.emission_velocity_variance = torch.rand((self.emission_dim, self.emission_dim))
        self.emission_dim *= 2
        self.n_parameters += self.emission_velocity_variance.numel()

    def _construct_emission_matrix(self):
        return torch.eye(self.emission_dim)

    def _construct_emission_covariance(self):
        emv = self.emission_velocity_variance @ self.emission_velocity_variance.T
        R = []
        for ac in self.approximate_covariance:
            emit = torch.zeros((
                len(ac), self.emission_dim, self.emission_dim
            ))
            emit[:,:self.latent_dim,:self.latent_dim] = emv
            emit[:,self.latent_dim:,self.latent_dim:] = ac
            R.append(emit)
        return R

    def build_batch_parameters(self, batch: int) -> LDSParameters:
        params = super().build_batch_parameters(batch)
        params.emission_covariance = self.global_parameters.emission_covariance[batch]
        return params

    def _calculate_sufficient_statistics(self, values: MomentumResuts) -> KalmanStatistics:
        r"""Calculate sufficient statistics for the momentum with velocity model.
        Since our log-likelihood calculation has non-constant emissions,
        we calculate $\mathbb{E}[v_t v_t^T]$, $\mathbb{E}[v_t \hat{v}_t^T]$, $\mathbb{E}[\hat{v}_tv_t^T]$, and $\mathbb{E}[\hat{v}_t\hat{v}_t^T]$.
        We store $\mathbb{E}[\hat{v}_t\hat{v}_t^T]$ in KalmanStatistics.Ez1z since it's unused and we 
        still want to keep stats.Ezz from the momentum model for our latent storage.

        """
        stats = super()._calculate_sufficient_statistics(values)

        Exx,Exz,Ezx,Ezz = [],[],[],[]
        for sm,x in zip(values.smoothed_mean, self.velocity):
            sm = sm[:,:self.latent_dim]
            exx = x @ x.mT 
            exz = x @ sm.mT 
            ezx = exz.mT
            ezz = sm @ sm.mT

            Exx.append(exx)
            Exz.append(exz)
            Ezx.append(ezx)
            Ezz.append(ezz)

        stats.Exx = Exx 
        stats.Ezx = Ezx
        stats.Exz = Exz
        stats.Ez1z = Ezz
        return stats

    def _solve_parameters(
            self, 
            values: MomentumResults, 
            stats: SufficientStatistics, 
            optimizer: str = "Adam", 
            lr: float = 0.01, 
            n_epochs: int = 1000, 
            gd_tol: float = 0.001
        ) -> torch.Tensor:
        r"""Solve for the parameters $\sigma$, $\lambda$ and $\mathbf{K}$. 
        We solve for the two inner parameters using the same method as the 
        momentum model, so we copy those equations.
        However, for the velocity covariance, we have to include the observed velocity.
        We can decompose our log-gaussian into the velocity and position components
        since the emission covariance has no connection on the diagonals. In this case,
        the position component is a constant and the transition matrix is an identity
        matrix, so we are left with only the velocity component with the form:
        $$Q(\theta, \hat{\theta}) = -\frac{T}{2} ln |K| 
            - \frac{1}{2}\mathbb{E}\left[
                \sum_{t=1}^T (v_t - \hat{v}_t)^T K^{-1}
                    (v_t - \hat{v}_t)
            \right]
        $$
        """
        decay = hseu.grad_tensor(self.decay)
        diffusion = hseu.grad_tensor(self.diffusion)
        K = hseu.grad_tensor(self.emission_velocity_variance)
        params = [decay, diffusion, K]

        def loss_closure(
                params, 
                n_batches: int, 
                stats: SufficientStatistics,
            ):
            decay,diffusion,K = params

            Ezz = stats.Ezz
            Ezz1 = stats.Ezz1
            
            # 2D for the emission log-likelihood
            EzzT = stats.Ez1z
            Ezx = stats.Ezx
            Exz = stats.Exz
            Exx = stats.Exx

            lmb = torch.exp(decay)
            sig = torch.exp(diffusion)
            M = 1 - lmb * self.dt 
            sigma = sig**2 * self.dt
            v0 = sigma / (1 - M**2)
            Kt = K @ K.T

            btrace = torch.vmap(torch.trace)

            total_loss = 0
            for i in range(n_batches):
                T = len(Ezz[i])

                iloss = Ezz[i][0] / v0
                iloss = self.latent_dim * torch.log(v0) + iloss 

                hloss = Ezz[i][1:] - 2 * M * Ezz1[i] + M**2 * Ezz[i][:-1]
                hloss = torch.sum(hloss, axis=0) / sigma 
                hloss = self.latent_dim * (T-1) * torch.log(sigma) + hloss

                eloss = Exx[i] - Ezx[i] - Exz[i] + EzzT[i]
                eloss = hseu.mulinv(Kt, eloss)
                eloss = btrace(eloss) 
                eloss = torch.sum(eloss, axis=0)
                eloss = T * torch.logdet(Kt) + eloss

                total_loss += (iloss + hloss + eloss) / 2

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

        self.decay = params[0].detach()
        self.diffusion = params[1].detach()
        self.emission_velocity_variance = params[2].detach()

        self._initialize_globals()
        return loss

    def fit(self, 
            X: list[np.ndarray|torch.Tensor]|None = None, 
            n_iter: int = 1000, 
            emtol: float = 1e-3, 
            **maximization_args
        ) -> MomentumResults:
        """Perform EM to fit the parameters.

        Args:
            X (list[np.ndarray|torch.Tensor]): If using 'observed' velocity, set this to None.
                If using 'true' velocity, a list of (T,ndims,1) arrays containing the velocity at each point.
            n_iter (int): Maximum number of iterations for which to run EM
            emtol (float): Minimum change in log-likelihood required for convergence.
            maximization_args: Passed to `_solve_parameters`
        
        Returns:
            MomentumResults: Fitted model information.
        """

        if self.velocity_type == 'true':
            X = self._initialize_observations(X)
            self.velocity = X
        elif self.velocity_type == 'observed':
            if self.emission_dim == 4:
                velocity = [
                    (
                        hseu.calculate_velocity_dt(
                            x[:,0].numpy().squeeze(),
                            self.dt.numpy()
                        ),
                        hseu.calculate_velocity_dt(
                            x[:,1].numpy().squeeze(),
                            self.dt.numpy()
                        )
                    )
                    for x in self.approximate_mean
                ]
                self.velocity = [
                    torch.hstack((
                        torch.from_numpy(v[0][:,None]), 
                        torch.from_numpy(v[1][:,None])
                    ))[...,None]
                    for v in velocity
                ]
            else:
                velocity = [
                    hseu.calculate_velocity_dt(
                        x[:,0].numpy().squeeze(),
                        self.dt.numpy()
                    )
                    for x in self.approximate_mean
                ]
                self.velocity = [
                    torch.from_numpy(v)[...,None]
                    for v in velocity
                ]
        else:
            raise ValueError(
                f'Unknown velocity type: {self.velocity_type}'
            )

        X = [
            torch.hstack((v,x)) for v,x in zip(
                self.velocity,
                self.approximate_mean
            )
        ]
        return LinearGaussianSystem.fit(
            self,
            X,
            n_iter,
            emtol,
            **maximization_args
        )