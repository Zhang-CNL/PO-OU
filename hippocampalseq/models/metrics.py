import numpy as np
import hippocampalseq.utils as hseu


def trajectory_error_centered(
        x: np.ndarray,
        x_hat: np.ndarray,
        t: np.ndarray,
        t_hat: np.ndarray,
        environment_size: list[tuple[int,...]],
        bin_size: int = 2.0
    ):
    mids = np.ndarray([
        int(round((es[1] - es[0]) / 2.0)) 
        for es in environment_size
    ])[:,None]
    decoded_data_size = (max(mids) * 2) + 1

    translated = translate_decoding(
        x_hat,
    )

def _interp(
        x: np.ndarray,
        x_hat: np.ndarray,
        t: np.ndarray,
        t_hat: np.ndarray
    ):
    mask = (t >= t_hat.min()) & (t <= t_hat.max())

    t_common = t[mask]
    x_common = x[mask]

    # Interpolate x_hat onto t_common
    if x.ndim == 1:
        x_hat_interp = np.interp(t_common, t_hat, x_hat)
    else:
        x_hat_interp = np.column_stack([
            np.interp(t_common, t_hat, x_hat[:, d])
            for d in range(x_hat.shape[1])
        ])

    return x_common, x_hat_interp

def trajectory_error_posterior(
        x: np.ndarray,
        xhat: np.ndarray,
        t: np.ndarray|None = None,
        that: np.ndarray|None = None
    ):
    """Computes the euclidean error between decoded position and true position.
    Interpolates points between whichever one is longer.

    Args:
        x (np.ndarray): True position
        xhat (np.ndarray): Decoded position
        t (np.ndarray): Time position corresponding to true position
        that (np.ndarray|None): Time corresponding to decoded position. If None, interpolates between true position times. Defaults to None.

    Returns:
        (np.ndarray): Error at each position.
        (np.ndarray): Mean error for all positions.
        (np.ndarray): Median error for all positions.

    """
    if that is None and t is not None:
        start,end = t[0],t[-1]
        sampling = (end - start) / len(xhat)
        that = np.arange(
            start,
            end,
            sampling
        )
        that = that[:len(xhat)]
    else:
        n = min(len(x), len(xhat))
        x = x[:n]
        xhat = xhat[:n]

    if len(x) > len(xhat):
        x_common,x_hat_interp = _interp(
            x.squeeze(),
            xhat.squeeze(),
            t.squeeze(), 
            that.squeeze()
        )
    elif len(xhat) > len(x):
        x_hat_interp,x_common = _interp(
            xhat.squeeze(),
            x.squeeze(),
            that.squeeze(),
            t.squeeze()
        )
    else:
        x_common,x_hat_interp = x,xhat

    position_error = np.sqrt((x_common - x_hat_interp)**2)
    mean_error = np.mean(position_error)
    median_error = np.median(position_error)
    return (
        position_error,
        mean_error,
        median_error
    )
