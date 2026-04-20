import numpy as np
import scipy.stats as sp

import hippocampalseq.utils as hseu

def __draw_clipped_normal_sample(mean, sd, low, high):
    counter = 0
    while True:
        sample = sp.multivariate_normal.rvs(
            mean, sd
        )
        if np.all(sample >= low) and np.all(sample < high):
            return sample
        counter += 1
        if counter > 200:
            print(mean, sd, sample)
            raise ValueError("Could not draw sample")

def simulate_momentum(n_steps, initial_diffusion_m, diffusion_m, decay, dt, seed=None):
    # meters to centimeters
    np.random.seed(seed)
    sd0 = initial_diffusion_m * 100 
    sd = diffusion_m * 100

    low = np.array([0,0])
    high = np.array([hseu.PFEIFFER_ENV_WIDTH_CM, hseu.PFEIFFER_ENV_HEIGHT_CM])

    trajectory = np.zeros((n_steps, 2))

    p0 = np.random.randint(low, high, size=2)
    trajectory[0] = p0

    p1 = __draw_clipped_normal_sample(p0, sd0**2 * dt * np.eye(2), low, high)
    trajectory[1] = p1

    m1 = (1 + np.exp(-decay * dt))
    m2 = np.exp(-decay * dt)
    var = (sd * dt)**2 / (2 * decay) * (1 - np.exp(-2 *decay * dt)) * np.eye(2)

    for t in range(2, n_steps):
        mean = m1 * trajectory[t-1] - m2 * trajectory[t-2]
        trajectory[t] = __draw_clipped_normal_sample(mean.ravel(), var, low, high)

    return trajectory
