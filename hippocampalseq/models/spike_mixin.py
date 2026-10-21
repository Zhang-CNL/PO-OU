import torch
import hippocampalseq.utils as hseu

class SpikeMixin:
    """Mixin with common functionality for various models that use spike data as observed variables."""
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
