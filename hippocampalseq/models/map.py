import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass

import hippocampalseq.utils as hseu
from .statespace import *


@dataclass
class BayesianMAPResults:
    decoded_trajectories: List[npt.ArrayLike]
    cumulative_probabilities: npt.ArrayLike

class BayesianMAP(StateSpace):
    def __init__(self, place_fields: npt.ArrayLike, dt: float, bin_size_cm: float):
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
            spikemat: npt.ArrayLike, 
            decoding_method: str = 'max'
        ) -> npt.ArrayLike:
        """MAP decoding of a single trajectory.

        Args:
            spikemat (npt.ArrayLike): Spikemat of shape (T, Ncell)
            decoding_method (str, optional): Decoding method. Can either take the maximum or the center of mass. Defaults to 'max'.

        Returns:
            npt.ArrayLike: Decoded trajectory of shape (T, 2)
        """
        assert decoding_method in ['max', 'center_of_mass']
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

        T,H,W = emission_probability.shape

        if decoding_method == 'max':
            indices = np.nanargmax(emission_probability.reshape(T,-1), axis=1)
            rows, cols = np.unravel_index(indices, (H, W))
        elif decoding_method == 'center_of_mass':
            yy, xx = np.indices((H, W))
            rows = np.sum(emission_probability * yy, axis=(1, 2)) / norm_factor
            cols = np.sum(emission_probability * xx, axis=(1, 2)) / norm_factor

        rows = rows * self.bin_size + self.bin_size / 2
        cols = cols * self.bin_size + self.bin_size / 2

        cum_prob = emission_probability.sum(axis=0)

        return np.column_stack((cols, rows)),cum_prob

    def fit(self, 
            X: List[npt.ArrayLike], 
            maximization_type: str = 'max',
            *_: Tuple[Any,...],
            **__: Dict[Any,Any]
        ) -> BayesianMAPResults:
        """Run MAP decoding on a list of spikemats.
        Args:
            X (List[npt.ArrayLike]): List of spikemats.
            maximization_type (str, optional): Decoding method. Can either take the 'max' or the 'center_of_mass'. Defaults to 'max'.

        Returns:
            BayesianMAPResults: Decoded trajectories and cumulative probabilities.
        """
        trajectories = []
        cum_probs    = np.zeros(self.place_fields.shape)
        for t,spike in enumerate(X):
            trajectory,cum_probs[t] = self.bayesian_decoding_one(spike, maximization_type)
            trajectories.append(trajectory)
        return BayesianMAPResults(
            trajectories,
            cum_probs
        )


