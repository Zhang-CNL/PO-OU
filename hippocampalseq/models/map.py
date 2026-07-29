import numpy as np
from dataclasses import dataclass

import hippocampalseq.utils as hseu
from .statespace import *


@dataclass
class BayesianMAPResults:
    decoded_trajectories: list[np.ndarray]
    cumulative_probabilities: np.ndarray

class BayesianMAP(StateSpace):
    def __init__(self, place_fields: np.ndarray, dt: float, bin_size_cm: float):
        """Model for Bayesian Maximum A-Posteriori decoding.
        Args:
            place_fields (np.ndarray|torch.Tensor): (Ncells, Nbx, Nby) Place field grids.
            dt (float): Time step for the transition matrix.
            bin_size_cm (float): Bin size in centimeters
        """
        self.place_fields = place_fields
        self.dt = dt
        self.bin_size = bin_size_cm

    def bayesian_decoding_one(
            self,
            spikemat: np.ndarray, 
        ) -> np.ndarray:
        """MAP decoding of a single trajectory.

        Args:
            spikemat (npt.ArrayLike): Spikemat of shape (T, Ncell)

        Returns:
            npt.ArrayLike: Decoded trajectory of shape (T, 2)
        """
        spikemat_nonzero = spikemat[np.where(spikemat.sum(axis=1) > 0)]
        emission_probability = hseu.calc_poisson_emission_probabilities_2d(
            spikemat_nonzero, 
            self.place_fields, 
            self.dt
        )
        norm_factor           = emission_probability.sum(axis=(1, 2))
        emission_probability  = emission_probability[~(norm_factor == 0),...]
        norm_factor           = norm_factor[~(norm_factor == 0)]
        emission_probability /= norm_factor[:,None,None]

        T = emission_probability.shape[0]
        coords = emission_probability.squeeze().shape[1:]

        indices = np.nanargmax(emission_probability.reshape(T,-1), axis=1)
        max_coords = np.unravel_index(indices, coords)
        max_coords = np.column_stack(max_coords[::-1]) # Reverse from ij to xy
        

        max_coords = max_coords * self.bin_size + self.bin_size / 2
        cum_prob = emission_probability.sum(axis=0)

        return max_coords,cum_prob

    def fit(self, 
            X: list[np.ndarray], 
            *_: tuple,
            **__: dict
        ) -> BayesianMAPResults:
        """Run MAP decoding on a list of spikemats.
        Args:
            X (list[npt.ArrayLike]): List of spikemats.
            maximization_type (str, optional): Decoding method. Can either take the 'max' or the 'center_of_mass'. Defaults to 'max'.

        Returns:
            BayesianMAPResults: Decoded trajectories and cumulative probabilities.
        """
        trajectories = []
        cum_probs    = np.zeros((len(X),)+self.place_fields.shape[1:])
        for t,spike in enumerate(X):
            trajectory,cum_probs[t] = self.bayesian_decoding_one(spike)
            trajectories.append(trajectory)
        return BayesianMAPResults(
            trajectories,
            cum_probs
        )


