import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from dataclasses import dataclass
from .statespace import *
import hippocampalseq.utils as hseu

__all__ = [
    'GaussianApproximation',
    'GaussianApproximationResults'
]

@dataclass 
class GaussianApproximationResults:
    approximate_means        : list[np.ndarray]
    approximate_covariances  : list[np.ndarray]
    cumulative_probabilities : np.ndarray

class GaussianApproximation(StateSpace):
    """Trajectory decoding by treating each individual time point
    as a Gaussian. Estimates the mean and covariance indepdendently
    for each time bin.
    """
    def __init__(self, place_fields: np.ndarray, dt: float, bin_size: float, environment_size: list[tuple[int,...]]):
        self.place_fields = place_fields
        self.dt = dt
        self.bin_size = bin_size
        self.grid = hseu.make_ndgrid(environment_size, bin_size)

    def name(self):
        return "Gaussian Approximation"

    def approximate_spikemat(self, spikemat: np.ndarray):
        spikemat = spikemat[np.where(spikemat.sum(axis=1)) > 0]
        emission_probability = hseu.calc_poisson_emission_probabilities_2d(
            spikemat, 
            self.place_fields,
            self.dt
        )
        emission_probability /= np.sum(emission_probability, axis=(1,2), keepdims=True)
        emission_probability = np.nan_to_num(emission_probability, nan=0.0, posinf=0.0, neginf=0.0)

        approx_mean, approx_cov = hseu.analytical_gaussian_approximation(
            self.grid,
            hseu.ensure_torch(emission_probability),
        )
        approx_mean,approx_cov = approx_mean.numpy(), approx_cov.numpy()

        probs = np.zeros(emission_probability.shape)
        for t in range(len(approx_mean)):
            mu,cov = approx_mean[t],approx_cov[t]
            mvn = multivariate_normal(mu.squeeze(), cov)
            probs[t] = mvn.logpdf(self.grid).reshape(probs.shape[1:])
        probs       -= logsumexp(probs, axis=(1,2), keepdims=True)
        probs   = np.exp(probs)
        cumprob = np.sum(probs, axis=0)
        cumprob     /= np.sum(cumprob)

        return approx_mean, approx_cov, cumprob.T

    def fit(self,
        X: list[np.ndarray],
        *_: tuple,
        **__: dict
    ) -> GaussianApproximationResults:
        means = []
        covs = []
        cumprobs = []
        for spikes in X:
            mean,cov,cumprob = self.approximate_spikemat(spikes)
            means.append(mean)
            covs.append(cov)
            cumprobs.append(cumprob)

        return GaussianApproximationResults(
            means,
            covs,
            cumprobs
        )

    def transform(self,
        X: list[np.ndarray],
    ) -> GaussianApproximationResults:
        return self.fit(X)