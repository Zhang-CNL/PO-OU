import numpy as np
import pynapple as nap
from typing import Dict

def assign_spikes_theta_phase(
        spike_info: Dict[int, nap.TsdFrame], 
        lfp_with_cycles: nap.TsdFrame
    ) -> Dict[int, nap.TsdFrame]:
    """Takes the full spike information and uses the LFP information to assign the bins what phase of theta
    they're in.

    Args:
        spike_info (Dict[int,nap.TsdFrame]): A dict matching cell IDs to their information.
        lfp_with_cycles (nap.TsdFrame): The full LFP data after `hippocampalseq.preprocessing.theta.detect_theta_cycles` has been called on it.
        
    Returns:
        (Dict[int,nap.TsdFrame]): Cell ID matched to the frame with theta pahse assigned.
    """

    # TODO: Check to see if the cell type field is used anywhere else
    lfp_times   = lfp_with_cycles['Phase Deg'].index.values.astype(float)
    phase_deg   = np.asarray(lfp_with_cycles['Phase Deg'].values, dtype=float)
    cycle_dur_s = np.asarray(lfp_with_cycles['Cycle Duration'].values, dtype=float)
    monotonic   = np.asarray(lfp_with_cycles['Monotonic Increasing'].values, dtype=float)


    def nearest_indices(query_times):
        q     = np.asarray(query_times, dtype=float)
        q     = np.clip(q, lfp_times[0], lfp_times[-1])
        right = np.searchsorted(lfp_times, q, side="left")
        right = np.clip(right, 0, len(lfp_times) - 1)
        left  = np.clip(right - 1, 0, len(lfp_times) - 1)
        pick_left = (np.abs(q - lfp_times[left]) <= np.abs(lfp_times[right] - q))
        return np.where(pick_left, left, right)

    out       = {}
    all_diffs = []

    for cell_id in spike_info.keys():
        ts = spike_info[cell_id]
        if len(ts) == 0:
            continue

        spike_times = ts.index.values.astype(float)
        idx = nearest_indices(spike_times)
        lfp_match_times = lfp_times[idx]

        # Pull columns by position 
        cols      = list(ts.columns)
        x_vals    = ts['x'].values
        y_vals    = ts['y'].values
        vel_vals  = ts['Velocity'].values
        time_diff = ts['Delta t'].values
        #cell_type_a = ts.values[:, cols.index('cell_type')]  # already an array

        out[cell_id] = nap.TsdFrame(
            t=spike_times,
            d=np.c_[
                x_vals, y_vals, vel_vals, time_diff, #cell_type_a,
                phase_deg[idx], cycle_dur_s[idx], monotonic[idx],
                lfp_match_times,
            ],
            columns=[
                'x', 'y', 'Velocity', 'Delta t', #'cell_type',
                'Phase Deg', 'Cycle Duration', 'Monotonic Increasing',
                'LFP Sample Time',
            ],
        )
        all_diffs.append(np.abs(spike_times - lfp_match_times))

    if all_diffs:
        td = np.concatenate(all_diffs)
        print(f"Matched {td.size} spikes to LFP. |Δt| (median/IQR): "
            f"{np.median(td)*1000:.2f} ms / "
            f"{(np.percentile(td,75)-np.percentile(td,25))*1000:.2f} ms")

    return out
