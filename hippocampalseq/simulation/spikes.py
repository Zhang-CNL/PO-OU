import numpy as np

import hippocampalseq.utils as hseu

def simulate_spikes(trajectory, place_fields, dt, bin_size, seed=None):
    np.random.seed(seed)
    n_cells = len(place_fields)

    spikes = np.zeros((len(trajectory), n_cells))

    for t, (x,y) in enumerate(trajectory):
        x = int(hseu.cm_to_bins(x, bin_size))
        y = int(hseu.cm_to_bins(y, bin_size))
        place_field = place_fields[:,x,y]
        spikes[t] = np.random.poisson(place_field * dt, size=n_cells)

    return spikes