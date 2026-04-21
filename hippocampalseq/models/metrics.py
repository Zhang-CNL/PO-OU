import numpy as np

def trajectory_error_posterior(true_trajectectory, estimated_trajectory):
    err = np.sqrt(
        np.sum((true_trajectectory - estimated_trajectory)**2, axis=1)
    )
    err = np.sort(err)
    cum_prob = np.linspace(0,1,len(err))  
    cum_error = np.column_stack((err, cum_prob))
    return cum_error