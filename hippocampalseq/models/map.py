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
    def bayesian_decoding_one(place_fields: npt.ArrayLike, spikemat: npt.ArrayLike, dt):
        emission_probability = hseu.calc_poisson_emission_probabilities_2d(spikemat, place_fields, dt)
        trajectory = np.empty(shape=(spikemat.shape[0], 2))
        for i,ep in enumerate(emission_probability):
            trajectory[i] = np.unravel_index(np.argmax(ep), ep.shape)
        return trajectory

    def fit(self, 
            X: List[npt.ArrayLike], 
            n_iter: int = 100, 
            emtol: float = 1e-3, 
            maximization_type: str = 'em'
        ) -> BayesianMAPResults:
        trajectories = []
        for spike in X:
            trajectory = self.bayesian_decoding_one(self.place_fields, spike, self.dt)
            trajectories.append(trajectory)
        return BayesianMAPResults(trajectories)


