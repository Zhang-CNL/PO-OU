import numpy as np
import pynapple as nap

def calculate_theta_oscillation_props(
        position_data: nap.TsdFrame,
        spike_data: dict[int,nap.TsdFrame],
        lfp_data: nap.TsdFrame,
        excitatory_neurons: np.ndarray,
        decoding_time_window: float = 0.02,
        decoding_time_advance: float = 0.005,
        theta_length_s: tuple[float,float] = (0.08,0.16),
        velocity_cutoff: float = 5.0,
        major_peak_bimodal_window: tuple[int,int] = (200,70),
        minor_peak_bimodal_window: tuple[int,int] = (80,190)
    ):
    theta_seq_start_phase = major_peak_bimodal_window[1]

