from dataclasses import dataclass
from typing import List, Tuple, Any, Dict
import numpy.typing as npt
import numpy as np
from scipy.special import logsumexp

import hippocampalseq.utils as hseu
from .statespace import *

@dataclass
class StationaryResults:
    model_evidence: List[float]
    cumulative_probabilities: npt.ArrayLike

class Stationary(Statespace):
    def __init__(self, place_fields: npt.ArrayLike, dt: float):
        super().__init__()

        self.place_fields = place_fields
        self.dt = dt

    def fit(self,
        X: List[npt.ArrayLike],
        *_: Tuple[Any,...],
        **__: Dict[Any,Any]
    ):
        cumulative_probabilities = np.zeros_like((len(X),)+self.place_fields.shape[1:])
        model_evidence = []
        for t,spike in enumerate(X):
            spike = spike[np.where(spike.sum(axis=1))]
            epl = hseu.calc_poisson_emission_probabilities_log_2d(
                spike,
                self.place_fields,
                self.dt
            )
            joint_prob = epl - np.log(epl.shape[1] * epl.shape[2])
            evidence = logsumexp(joint_prob)
            marginals = np.exp(epl - epl.max())

            model_evidence.append(evidence)
            cumulative_probabilities[t] = marginals

        return StationaryResults(
            model_evidence,
            cumulative_probabilities
        )