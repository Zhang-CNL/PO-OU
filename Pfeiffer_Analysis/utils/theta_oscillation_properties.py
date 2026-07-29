
import pynapple as nap
import numpy as np
import pathlib as Path
import matplotlib.pyplot as plt

def pre_decoding_check(position_data, spike_data, 
                        lfp_data, excitatory_neurons,
                        bimodal_windows,
                        initial_variables,
                        vel_cut=10):
    """
    Mimics MATLAB IRFS_CALCULATE_THETA_OSCILLATION_PROPERTIES 
    which basically just sets up the decoding windows prior to running the bayesian decoder

    Builds Decoding_Time_Info (DTI), spike index matrix, and valid window index.
    """

    # Parameters defined by Brad
    time_win  = initial_variables['decoding_time_window']     # 0.02
    time_step = initial_variables['decoding_time_advance']    # 0.005
    th_min, th_max = initial_variables['theta_length_min_max']  # (0.08, 0.16)
    ## These are starting windows and subject to change after decoding
    major_win = np.asarray(bimodal_windows['major_peak_window'])  # [200, 70] (wrap)
    minor_win = np.asarray(bimodal_windows['minor_peak_window'])  # [80, 190]

    theta_sequence_starting_phase = 70.0

    # 
    # Load LFP, position, spike data
    # 
    phase_ts  = lfp_data['Phase Deg']
    lfp_t     = phase_ts.index.values
    lfp_phase      = phase_ts
    lfp_power = np.asarray(lfp_data['Power'].values)

    pos_t = position_data.index.values
    x     = position_data['x'].values
    y     = position_data['y'].values
    vel   = position_data['Velocity'].values
    #mdir  = position_data['movement_direction'].values
    hdg   = position_data['Head direction'].values

    # print("type(spike_data):", type(spike_data))
    exc_cell_ids   = [cid for cid in spike_data.keys() if cid in excitatory_neurons]
    spike_data_exc = {cid: spike_data[cid] for cid in exc_cell_ids}
    exc_with_spikes = [cid for cid in exc_cell_ids if len(spike_data_exc[cid]) > 0]
    empty_exc = [cid for cid in spike_data.keys()
                if cid in excitatory_neurons and len(spike_data[cid]) == 0]
    # print(f"Excitatory cells with no spikes after cleaning: {empty_exc}")
    # print(f"Excitatory cells used for decoding: {len(exc_with_spikes)}")

    spike_start = min(spike_data_exc[cid].index[0]  for cid in exc_with_spikes)
    spike_end   = max(spike_data_exc[cid].index[-1] for cid in exc_with_spikes)

    start_time = max(pos_t[0], lfp_t[0], spike_start)
    end_time   = min(pos_t[-1], lfp_t[-1], spike_end)

    decoding_times = np.arange(
        start_time + time_win/2.0,
        end_time   - time_win/2.0 + 1e-12,
        time_step
    )

    # Restrict to clean intervals (drops noisy-epoch and session-gap times)
    ##This should technically already be done in the preprocessing but Brad does have it here
    clean_intervals = lfp_data['run_interval']
    keep = np.zeros(len(decoding_times), dtype=bool)
    for s, e in zip(clean_intervals.start, clean_intervals.end):
        keep |= (decoding_times >= s) & (decoding_times <= e)
    decoding_times = decoding_times[keep]

    # 
    # Flatten excitatory spikes
    # 
    exc_times, exc_cells = [], []
    for cid in exc_with_spikes:
        t = spike_data_exc[cid].index.values.astype(float)
        if t.size == 0:
            continue
        exc_times.append(t)
        exc_cells.append(np.full(t.size, cid, dtype=int))

    if len(exc_times) == 0:
        exc_times = np.array([], dtype=float)
        exc_cells = np.array([], dtype=int)
    else:
        exc_times = np.concatenate(exc_times)
        exc_cells = np.concatenate(exc_cells)
        order = np.argsort(exc_times)
        exc_times = exc_times[order]
        exc_cells = exc_cells[order]

    # 
    # Build DTI
    # 
    nW = decoding_times.size
    DTI = np.zeros((nW, 14))

    # Nearest pos/LFP indices for window centers
    pos_idx = np.searchsorted(pos_t, decoding_times, side='left')
    pos_idx = np.clip(pos_idx, 0, len(pos_t) - 1)
    back = np.clip(pos_idx - 1, 0, len(pos_t) - 1)
    pick_back = (np.abs(decoding_times - pos_t[back])
                <= np.abs(pos_t[pos_idx] - decoding_times))
    pos_idx = np.where(pick_back, back, pos_idx)

    lfp_idx = np.searchsorted(lfp_t, decoding_times, side='left')
    lfp_idx = np.clip(lfp_idx, 0, len(lfp_t) - 1)
    back = np.clip(lfp_idx - 1, 0, len(lfp_t) - 1)
    pick_back = (np.abs(decoding_times - lfp_t[back])
                <= np.abs(lfp_t[lfp_idx] - decoding_times))
    lfp_idx = np.where(pick_back, back, lfp_idx)

    DTI[:, 0]  = x[pos_idx]
    DTI[:, 1]  = y[pos_idx]
    DTI[:, 2]  = hdg[pos_idx] if hdg is not None else np.nan
    DTI[:, 3]  = vel[pos_idx]
    #DTI[:, 4]  = mdir[pos_idx]
    DTI[:, 5]  = lfp_phase[lfp_idx]
    DTI[:, 10] = lfp_power[lfp_idx]

    # 
    # Spike counts per window + spike index matrix
    # 
    max_per_win = 200
    spike_index_mat = np.zeros((max_per_win, nW), dtype=int)

    half = time_win / 2.0
    s_left  = np.searchsorted(exc_times, decoding_times - half, side='left')
    s_right = np.searchsorted(exc_times, decoding_times + half, side='right')

    for i in range(nW):
        if exc_times.size == 0:
            continue
        lo, hi = s_left[i], s_right[i]
        if hi <= lo:
            continue
        cids = exc_cells[lo:hi]
        DTI[i, 6] = cids.size
        DTI[i, 7] = np.unique(cids).size
        if cids.size > 0:
            spike_index_mat[:cids.size, i] = cids

    # Trim spike index matrix
    if nW > 0:
        nonzero_rows = np.where(np.max(spike_index_mat, axis=1) > 0)[0]
        if nonzero_rows.size > 0:
            spike_index_mat = spike_index_mat[:nonzero_rows[-1] + 1, :]
        else:
            spike_index_mat = spike_index_mat[:0, :]

    # 
    # Oscillation segmentation
    # 
    # Detect session boundaries: adjacent decoding_times rows separated by
    # more than ~1 time_step (i.e., spanning a removed gap or a session join)
    # Tthis is because I initially did not break up the sessions that have the animal running then put back in the home cage.  
    # Since they use the same recording, I treated it as theh same session but concatenated position and lfp data
    dt_arr = np.diff(decoding_times)
    boundary_pair = dt_arr > (1.5 * time_step)
    session_break_idx = np.where(boundary_pair)[0] + 1

    # Phase crossings, but ONLY between temporally-adjacent rows.
    # A "crossing" that straddles a session gap is an artifact of stitching.
    ph = DTI[:, 5].copy()
    raw_phase_cross = ((ph[:-1] < theta_sequence_starting_phase)
                    & (ph[1:] >= theta_sequence_starting_phase))
    valid_phase_cross = raw_phase_cross & ~boundary_pair
    phase_cross_idx = np.where(valid_phase_cross)[0] + 1

    # Union of phase crossings and forced breaks at session boundaries
    osc_starts = np.unique(np.concatenate([phase_cross_idx, session_break_idx]))
    osc_ends   = (np.concatenate([osc_starts[1:], [nW]])
                if osc_starts.size > 0 else np.array([], dtype=int))
    n_osc = osc_starts.size

    # Assign oscillation IDs (rows before osc_starts[0] keep id=0, like before)
    osc_id = np.zeros(nW, dtype=int)
    for k in range(n_osc):
        osc_id[osc_starts[k]:osc_ends[k]] = k + 1
    DTI[:, 13] = osc_id

    print(f"  Detected oscillations: {n_osc}")
    print(f"  Session-boundary breaks added: {session_break_idx.size}")
    print(f"  Phase-crossings rejected at boundaries: "
        f"{int((raw_phase_cross & boundary_pair).sum())}")

    # 
    # Per-oscillation flags + major/minor spike counts
    # Iterates over contiguous slices instead of scanning the full osc_id array
    # for each oscillation. O(n_osc * cycle_len) instead of O(n_osc * nW).
    # 
    major_start, major_end = major_win
    minor_start, minor_end = minor_win

    for k in range(n_osc):
        a, b = osc_starts[k], osc_ends[k]
        cycle_len = b - a
        if cycle_len == 0:
            continue
        cs = slice(a, b)

        # Velocity throughout the oscillation
        vel_ok = float(np.all(DTI[cs, 3] >= vel_cut))
        DTI[cs, 8] = vel_ok

        # Duration: within an oscillation we've guaranteed contiguous time,
        # so cycle_len * time_step is exact
        cycle_dur = cycle_len * time_step

        # Monotonic phase (after wrapping <=70° up to >=360°)
        if cycle_len > 1:
            ph_cycle = DTI[cs, 5].copy()
            ph_cycle[ph_cycle <= theta_sequence_starting_phase] += 360.0
            monotonic = bool(np.all(np.diff(ph_cycle) >= 0))
        else:
            monotonic = True

        dur_ok = bool(th_min <= cycle_dur <= th_max)
        DTI[cs, 9] = 1.0 if (monotonic and dur_ok) else 0.0

        # Major / minor spike counts (combined into same loop)
        phases_in_cycle = DTI[cs, 5]
        spikes_in_cycle = DTI[cs, 6]

        if major_end < major_start:
            major_mask = ((phases_in_cycle >= major_start)| (phases_in_cycle < major_end))
        else:
            major_mask = ((phases_in_cycle >= major_start) & (phases_in_cycle < major_end))
        DTI[cs, 11] = spikes_in_cycle[major_mask].sum()

        if minor_end < minor_start:
            minor_mask = ((phases_in_cycle >= minor_start)| (phases_in_cycle < minor_end))
        else:
            minor_mask = ((phases_in_cycle >= minor_start)& (phases_in_cycle < minor_end))
        DTI[cs, 12] = spikes_in_cycle[minor_mask].sum()

    # 
    # Valid windows: spikes>0 & velocity_ok & cycle_ok
    # 
    decoding_window_index = np.where(
        (DTI[:, 6] > 0) & (DTI[:, 8] == 1) & (DTI[:, 9] == 1)
    )[0]
    print(f"  Valid decoding windows: {decoding_window_index.size} / {nW}")

    predecode = {
        "DTI": DTI,
        "decoding_window_index": decoding_window_index,
        "decoding_spike_index": spike_index_mat,
        "decoding_times": decoding_times,
    }
    return predecode