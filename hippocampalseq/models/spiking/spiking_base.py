import torch
import hippocampalseq.utils as hseu
from functools import wraps
from torch.distributions import MultivariateNormal
from hippocampalseq.models.linear_gaussian_system import *
from dataclasses import dataclass, field

__all__ = [
    'SpikingBase',
    'count_spikes',
    'lag_spikes'
]

def count_spikes(spikemat: torch.Tensor) -> torch.Tensor:
    spk = hseu.ensure_torch(spikemat).double()
    return hseu.atleast_3d(spk.sum(dim=1))

def lag_spikes(spikecount: torch.Tensor, n_lag: int=1) -> torch.Tensor:
    return torch.roll(spikecount, shifts=n_lag, dims=0)

class SpikingBase(LinearGaussianSystem):
    def __init__(self,
            order: int,
            position_dims: slice,
            dt: float,
            environment_size: list[tuple[int,...]],
            bin_size: int,
            place_fields: hseu.NDArray,
            spikemat_train: list[hseu.NDArray],
            spikemat_valid: list[hseu.NDArray]|None=None,
            initialization_method: str|dict[str, hseu.NDArray] = 'uniform'
        ):

        if place_fields.shape[-1] == 1:
            nzdim = 1
        else:
            nzdim = 2

        super().__init__(
            nzdim,
            nzdim,
            order,
        )

        self.bin_size = bin_size
        self.environment_size = environment_size
        self.dt = torch.tensor(dt)
        self.grid = hseu.make_ndgrid(self.environment_size, self.bin_size, indexing='ij')

        self.place_fields = hseu.ensure_torch(place_fields)
        (
            self.emission_probabilities,
            self.approximate_mean,
            self.approximate_covariance
        ) = self.spike_approximations(spikemat_train)

        (
            self.validation_emission_probabilities,
            self.validation_approximate_mean,
            self.validation_approximate_covariance
        ) = self.spike_approximations(spikemat_valid)

        self.marginal_position_dims = position_dims


    def name(self):
        return "Spiking Linear Gaussian Model Base"

    def __str__(self):
        return self.name()

    def spike_approximations(self, spikemats: hseu.NDArray|None):
        if spikemats is None:
            return ([],[],[])
        emission_probabilities = []
        approximate_mean       = []
        approximate_covariance = []
        for spk in spikemats:
            (
                emission_probability, 
                approx_mean, 
                approx_cov
            ) = self.spike_to_gaussian(spk, self.place_fields, self.dt, self.grid)
            
            emission_probabilities.append(emission_probability)
            approximate_mean.append(approx_mean)
            approximate_covariance.append(approx_cov)
        return (
            emission_probabilities,
            approximate_mean,
            approximate_covariance
        )

    def _construct_emission_covariance(self, mode: str = "train") -> torch.Tensor:
        if mode == "train":
            return self.approximate_covariance
        elif mode == "valid":
            return self.validation_approximate_covariance
        elif mode == "test":
            return self.testing_approximate_covariance
        else:
            raise ValueError("Mode must be train, valid, or test")

    def build_batch_parameters(self, batch: int, mode: str = "train") -> LDSParameters:
        """Builds a single batch of parameters.
        In this case, only the emission covariance is treated as being batch dependent.
        If you want other variables to depend on the batch, do that yourself in a subclass.
        """
        return LDSParameters(
            transition_matrix     = self.global_parameters.transition_matrix,
            transition_covariance = self.global_parameters.transition_covariance,
            transition_bias       = self.global_parameters.transition_bias,
            emission_matrix       = self.global_parameters.emission_matrix,
            emission_covariance   = self.global_parameters.emission_covariance[batch],
            emission_bias         = self.global_parameters.emission_bias,
            initial_mean          = self.global_parameters.initial_mean,
            initial_covariance    = self.global_parameters.initial_covariance
        )

    def spike_to_gaussian(self, spikemat: hseu.NDArray, place_fields: torch.Tensor, dt: torch.Tensor|float, grid: torch.Tensor):
        spikemat = hseu.ensure_torch(spikemat).double()

        emission_probability = hseu.calc_poisson_emission_probabilities_2d(
            spikemat, 
            place_fields,
            dt
        )
        emission_probability /= torch.sum(emission_probability, axis=(1,2), keepdim=True)
        emission_probability = torch.nan_to_num(emission_probability, nan=0.0, posinf=0.0, neginf=0.0)

        approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
            grid,
            emission_probability,
            'weighted',
        )
        return emission_probability, approx_mean, approx_cov

    def _calculate_marginals(self, environment_size: list[tuple[int,...]], bin_size: int, values: KalmanResults) -> torch.Tensor:
        r"""Calculates the marginal probabilities for each bin in the environment.
        What is the probability that the mouse is in a given bin at a given time $P(X_t = x, Y_t = y|\mu_t, \Sigma_t)$

        Args:
            environment_size (list[tuple[int,...]]): List of tuples of size (min,max) for each axis.
            bin_size (int): Size of individual bins in cm.
            values (KalmanResults): Kalman filter results.

        Returns:
            torch.Tensor: The marginal probabilities for each bin in the environment. (Ncells, nbx, nby)
        """
        sz = tuple(int((es[1] - es[0]) / bin_size) for es in environment_size)
        if len(sz) == 1:
            sz = sz + (1,)
        Z = hseu.make_ndgrid(environment_size, bin_size, indexing='ij')
        cumulative_probabilities = torch.zeros((len(values.smoothed_mean),) + sz)

        for i in range(len(values.smoothed_mean)):
            sm = values.smoothed_mean[i][:,self.marginal_position_dims]
            sc = torch.atleast_2d(values.smoothed_cov[i][:,self.marginal_position_dims,self.marginal_position_dims])
            cp = torch.zeros((sm.shape[0],)+sz)
            for t in range(sm.shape[0]):
                cov_t = sc[t]
                try:
                    L = torch.linalg.cholesky(cov_t)
                except Exception as e:
                    print(i,t)
                    print(values.smoothed_cov[i][t-2:t+3])
                    print(values.filtered_cov[i][t-2:t+3])
                    print(values.predicted_cov[i][t-2:t+3])
                    print(values.smoothed_gain[i][t-2:t+3])
                    raise e

                mvn = MultivariateNormal(
                    sm[t].ravel(), 
                    scale_tril=L
                )
                log_prob = mvn.log_prob(Z)
                log_prob = log_prob.reshape(sz)
                cp[t] = log_prob
            
            cp -= torch.logsumexp(cp, dim=(1, 2), keepdim=True)
            cp = torch.exp(cp)
            cumulative_probabilities[i] = torch.sum(cp,axis=0)

        return cumulative_probabilities / cumulative_probabilities.sum(axis=(1, 2), keepdim=True)

    def fit(self,
            Xtrain: list[hseu.NDArray]|None=None,
            Xvalid: list[hseu.NDArray]|None=None,
            n_iter: int = 1000,
            emtol: float = 1e-3,
            patience: int = 5,
            val_freq: int = 10,
            **maximization_args
        ) -> LDSResults:
        """Run the Expectation-Maximization algorithm to fit the model parameters to the data.

        Parameters:
            Xtrain (list[hseu.NDArray]|None): If None, uses `self.approximate_mean` as the observed training values. Defaults to None.
            Xvalid (list[hseu.NDArray]|None): If None, uses `self.validation_approximate_mean` as observed validation. Defaults to None.
            n_iter (int): Maximum number of EM iterations. Defaults to 1000.
            emtol (float): Tolerance for the change in log-likelihood between iterations. Defaults to 1e-3.
            patience (int): Number of validation iterations without improvement before stopping. Defaults to 5.
            val_freq (int): Frequency of validation checks. Defaults to 10.
            **maximization_args: Keyword arguments to pass to the parent class's maximization method.

        Returns:
            LDSResults: Results of fitting the model to the data.
        """

        if Xtrain is None:
            Xtrain = self.approximate_mean
        if Xvalid is None and len(self.validation_approximate_mean) > 0:
            Xvalid = self.validation_approximate_mean

        results = super().fit(
            Xtrain,
            Xvalid,
            n_iter,
            emtol,
            patience,
            val_freq,
            **maximization_args
        )
        if not isinstance(results, tuple):
            results.cumulative_probabilities = self._calculate_marginals(
                self.environment_size,
                self.bin_size,
                results
            )
        else:
            for res in results:
                res.cumulative_probabilities = self._calculate_marginals(
                    self.environment_size,
                    self.bin_size,
                    res
                )       
        return results

    def transform(self,
            spikemats: list[hseu.NDArray],
            X: list[hseu.NDArray]|None = None
        ) -> LDSResults:
        (
            self.testing_emission_probabilities,
            self.testing_approximate_mean,
            self.testing_approximate_covariance
        ) = self.spike_approximations(spikemats)

        if X is None:
            X = self.testing_approximate_mean
        return super().transform(
            X
        ) 

