import numpy as np
import numpy.typing as npt
from typing import List
from dataclasses import dataclass

import hippocampalseq.utils as hseu
from .statespace import *


@dataclass
class BayesianMAPResults:
    trajectories: List[npt.ArrayLike]

class BayesianMAP(StateSpace):
    def __init__(self, place_fields: npt.ArrayLike, dt: float):
        self.place_fields = place_fields
        self.dt = dt

    @staticmethod
    def bayesian_decoding_one(
            place_fields: npt.ArrayLike,
            spikemat: npt.ArrayLike, 
            dt: npt.ArrayLike|float,
            decoding_method: str = 'max'
        ) -> npt.ArrayLike:
        assert decoding_method in ['max', 'center_of_mass']
        emission_probability = hseu.calc_poisson_emission_probabilities_2d(spikemat, place_fields, dt)
        if decoding_method == 'max':
            N,R,C = emission_probability.shape
            indices = np.argmax(emission_probability.reshape(N,-1), axis=1)
            rows = indices // C
            cols = indices % C
        elif decoding_method == 'center_of_mass':
            T, H, W = emission_probability.shape
            yy, xx = np.indices((H, W))

            rows = np.sum(emission_probability * yy, axis=(1, 2)) / np.sum(emission_probability, axis=(1, 2))
            cols = np.sum(emission_probability * xx, axis=(1, 2)) / np.sum(emission_probability, axis=(1, 2))

        return np.column_stack((rows, cols))

    def fit(self, 
            X: List[npt.ArrayLike], 
            decoding_method: str = 'max',
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'em'
        ) -> BayesianMAPResults:
        trajectories = []
        for spike in X:
            trajectory = self.bayesian_decoding_one(self.place_fields, spike, self.dt)
            trajectories.append(trajectory)
        return BayesianMAPResults(trajectories)


