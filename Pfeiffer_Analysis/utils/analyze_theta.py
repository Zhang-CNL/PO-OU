
import pynapple as nap
import numpy as np
import pathlib as Path
import matplotlib.pyplot as plt

def detect_theta_cycles(lfp_data, initial_variables=None,
                        plot=False, segment_seconds=5.0,
                        segment_start=None, random_segment=False,
                        max_cycle_duration=1.0):

    use_theta_filter = False
    theta_min, theta_max  = None, None
    if initial_variables is not None and \
        initial_variables.get('limit_analysis_by_theta_length', 0):
        theta_min, theta_max = initial_variables['theta_length_min_max']
        use_theta_filter = True
        #print(f'Theta length filter: {theta_min*1000:.0f}-{theta_max*1000:.0f} ms')

    # Get phase data
    phase_rad   = lfp_data['phase']
    phase_times = phase_rad.index.values
    phase_deg   = np.degrees(phase_rad.values) % 360

    #print(f'Phase range: {np.min(phase_deg):.1f} - {np.max(phase_deg):.1f} degrees')

    #  Detect troughs by phase resets (360 -> 0) 
    phase_diff = np.diff(phase_deg)
    troughs    = np.where(phase_diff < -345)[0] + 1

    n_samples            = len(phase_deg)
    cycle_duration       = np.zeros(n_samples)
    monotonic_increasing = np.zeros(n_samples)
    cycle_id             = np.zeros(n_samples, dtype=int)

    n_valid            = 0
    n_skipped_boundary = 0
    n_skipped_length   = 0

    for i in range(len(troughs) - 1):
        start_idx = troughs[i]
        end_idx   = troughs[i + 1]
        duration  = phase_times[end_idx] - phase_times[start_idx]

        # Always exclude cycles that span session boundaries
        if duration > max_cycle_duration:
            n_skipped_boundary += 1
            continue

        # Optionally exclude cycles outside the theta length range
        if use_theta_filter and (duration < theta_min or duration > theta_max):
            n_skipped_length += 1
            continue

        cycle_phases = phase_deg[start_idx:end_idx]
        is_monotonic = np.all(np.diff(cycle_phases) > 0)

        cycle_duration[start_idx:end_idx]       = duration
        monotonic_increasing[start_idx:end_idx] = 1 if is_monotonic else 0
        cycle_id[start_idx:end_idx]             = n_valid + 1
        n_valid += 1


    data_mat = np.column_stack([
        lfp_data['filtered_lfp'].values,
        lfp_data['amplitude'].values,
        phase_deg,
        cycle_duration,
        monotonic_increasing,
        cycle_id,
    ])
    cols = ['filtered_lfp', 'amplitude', 'phase_deg',
            'cycle_duration', 'monotonic', 'cycle_id']
    lfp_cycles = nap.TsdFrame(t=phase_times, d=data_mat,
                            columns=cols, time_units='s')

    out = dict(lfp_data)
    out['phase_deg']             = nap.Tsd(t=phase_times, d=phase_deg)
    out['cycle_duration']        = nap.Tsd(t=phase_times, d=cycle_duration)
    out['monotonic_increasing']  = nap.Tsd(t=phase_times, d=monotonic_increasing)
    out['cycle_id']              = nap.Tsd(t=phase_times, d=cycle_id)
    out['trough_indices']        = troughs
    out['trough_ts']             = nap.Ts(t=phase_times[troughs], time_units='s')
    out['amplitude']             = nap.Tsd(t=phase_times, d=lfp_data['amplitude'].values)
    out['power']                 = nap.Tsd(t=phase_times, d=lfp_data['power'].values)
    out['filtered_lfp_at_phase'] = nap.Tsd(t=phase_times, d=lfp_cycles['filtered_lfp'].values)

    valid_durations = cycle_duration[cycle_duration > 0]

    # Plotting
    figs = {}
    if plot:
        if segment_start is None:
            if random_segment:
                rng = np.random.default_rng(0)
                t0 = rng.uniform(
                    phase_times[0],
                    max(phase_times[-1] - segment_seconds, phase_times[0]),
                )
            else:
                t0 = phase_times[0]
        else:
            t0 = float(segment_start)
        t1 = min(t0 + segment_seconds, phase_times[-1])

        seg_mask = (phase_times >= t0) & (phase_times <= t1)
        t_seg    = phase_times[seg_mask]
        phi_seg  = phase_deg[seg_mask]
        lfp_seg  = lfp_data['filtered_lfp'].values[seg_mask]

        trough_t      = phase_times[troughs]
        trough_in_seg = (trough_t >= t0) & (trough_t <= t1)

        fig1, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax[0].plot(t_seg, lfp_seg, linewidth=0.5)
        ax[0].set_ylabel("Theta-filtered LFP (uV)")
        ax[0].set_title("Theta segment with trough markers")
        ax[0].vlines(trough_t[trough_in_seg],
                    ymin=np.min(lfp_seg), ymax=np.max(lfp_seg),
                    linestyles="dashed", alpha=0.6)
        ax[1].plot(t_seg, phi_seg, linewidth=0.5)
        ax[1].vlines(trough_t[trough_in_seg],
                    ymin=0, ymax=360, linestyles="dashed", alpha=0.6)
        ax[1].set_ylabel("Phase (deg)")
        ax[1].set_xlabel("Time (s)")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(valid_durations * 1000.0, bins=40)
        ax2.set_xlabel("Cycle duration (ms)")
        ax2.set_ylabel("Count")
        title = (f"Theta cycle duration distribution\n"
                f"{n_valid} valid, {n_skipped_boundary} boundary-rejected")
        if use_theta_filter:
            title += f", {n_skipped_length} length-filtered"
        ax2.set_title(title)
        if use_theta_filter:
            ax2.axvline(theta_min * 1000, color='red', linestyle='--',
                        alpha=0.5, label=f'{theta_min*1000:.0f} ms')
            ax2.axvline(theta_max * 1000, color='red', linestyle='--',
                        alpha=0.5, label=f'{theta_max*1000:.0f} ms')
            ax2.legend()

        figs = {'lfp_phase': fig1, 'cycle_durations': fig2}

    return out, figs