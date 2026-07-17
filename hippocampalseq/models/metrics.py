import numpy as np

def trajectory_error_posterior(
        true_trajectectory: np.ndarray, 
        estimated_trajectory: np.ndarray,
        environment_size: np.ndarray,
        bin_size_cm: float
    ):
    nbx = int((environment_size[2] - environment_size[0]) / bin_size_cm)
    nby = int((environment_size[3] - environment_size[1]) / bin_size_cm)
    sp_x = np.linspace(environment_size[0], environment_size[2], nbx + 1) + bin_size_cm / 2
    sp_y = np.linspace(environment_size[1], environment_size[3], nby + 1) + bin_size_cm / 2

    true_pos_hist,_,_ = np.histogram2d( 
        true_trajectectory[:,0], 
        true_trajectectory[:,1], 
        bins=(sp_x, sp_y)
    )
    true_pos_hist = true_pos_hist.T

    err = np.sqrt(
        np.sum((true_trajectectory - estimated_trajectory)**2, axis=1)
    )
    err = np.sort(err)
    cum_prob = np.linspace(0,1,len(err))  
    cum_error = np.column_stack((err, cum_prob))
    return cum_error