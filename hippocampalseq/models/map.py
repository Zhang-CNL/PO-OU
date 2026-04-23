import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Any
from dataclasses import dataclass

import hippocampalseq.utils as hseu
from .statespace import *


@dataclass
class BayesianMAPResults:
    trajectories: List[npt.ArrayLike]
    cumulative_probabilities: List[npt.ArrayLike]

class BayesianMAP(StateSpace):
    def __init__(self, place_fields: npt.ArrayLike, dt: float, bin_size_cm: float):
        self.place_fields = place_fields
        self.dt = dt
        self.bin_size = bin_size_cm

    def bayesian_decoding_one(
            self,
            spikemat: npt.ArrayLike, 
            decoding_method: str = 'max'
        ) -> npt.ArrayLike:
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
            decoding_method: str = 'max',
            *_: Tuple[Any,...]
        ) -> BayesianMAPResults:
        trajectories = []
        cum_probs    = []
        for spike in X:
            trajectory,cum_prob = self.bayesian_decoding_one(spike, decoding_method)
            trajectories.append(trajectory)
            cum_probs.append(cum_prob)
        return BayesianMAPResults(
            trajectories,
            cum_probs
        )


