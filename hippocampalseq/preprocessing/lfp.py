import numpy as np
import pynapple as nap

def assign_spikes_theta_phase(spike_info: nap.TsGroup, lfp_with_cycles):
    lfp_times   = lfp_with_cycles['phase'].index.values.astype(float)
    phase_deg   = np.asarray(lfp_with_cycles['phase_deg'].values, dtype=float)
    cycle_dur_s = np.asarray(lfp_with_cycles['cycle_duration'].values, dtype=float)
    monotonic   = np.asarray(lfp_with_cycles['monotonic_increasing'].values, dtype=float)


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
        x_vals    = ts.values[:, cols.index('x')]
        y_vals    = ts.values[:, cols.index('y')]
        vel_vals  = ts.values[:, cols.index('velocity')]
        time_diff = ts.values[:, cols.index('time_diff')]
        cell_type_a = ts.values[:, cols.index('cell_type')]  # already an array

        out[cell_id] = nap.TsdFrame(
            t=spike_times,
            d=np.c_[
                x_vals, y_vals, vel_vals, time_diff, cell_type_a,
                phase_deg[idx], cycle_dur_s[idx], monotonic[idx],
                lfp_match_times,
            ],
            columns=[
                'x', 'y', 'velocity', 'time_diff', 'cell_type',
                'theta_phase', 'cycle_duration', 'monotonic_increasing',
                'lfp_sample_time',
            ],
        )
        all_diffs.append(np.abs(spike_times - lfp_match_times))

    if all_diffs:
        td = np.concatenate(all_diffs)
        print(f"Matched {td.size} spikes to LFP. |Δt| (median/IQR): "
            f"{np.median(td)*1000:.2f} ms / "
            f"{(np.percentile(td,75)-np.percentile(td,25))*1000:.2f} ms")

    return out
