
from pathlib import Path
import scipy
import pynapple as nap
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, hilbert
from neo.io import NeuralynxIO
import struct
import os
import scipy.io

from utils.Data_Loading.vel_movement import calculate_movement_direction, calculate_velocity


def load_position_data(basepath, animal, experiment):
    
    data_dir = Path(basepath) / animal / experiment
    
    pos_file = data_dir / 'Position_Data.mat'
    pos_mat = scipy.io.loadmat(pos_file)
    raw_position = pos_mat['Position_Data']
    
    # Extract components
    time = raw_position[:, 0]
    x_pos = raw_position[:, 1]
    y_pos = raw_position[:, 2]
    head_direction = raw_position[:, 3]
    
    # Normalize positions to start at 0.001
    x_pos = x_pos - np.min(x_pos) + 0.001
    y_pos = y_pos - np.min(y_pos) + 0.001
    
    epoch_file = data_dir / 'Epochs.mat'
    pos_epoch = scipy.io.loadmat(epoch_file)
    #running_epoch = pos_epoch['Run_Times']
    #running_epoch = load_run_times(pos_epoch)
    #print(f"Position epoch shape: {np.shape(pos_epoch)}, with values: {pos_epoch}")
    rt = pos_epoch['Run_Times']
    # print(f"Running epoch shape: {np.shape(rt)}, with values: {rt}")
    # Squeeze away singleton dims
    rt = np.squeeze(rt)
    #print(f"Running epoch shape after squeeze: {np.shape(rt)}, with values: {rt}")

    # Now rt should be a numeric array with shape (N, 2) or (2,)
    #This takes into account the fact that some of the rats have multiple running epochs
    rt = np.atleast_2d(rt.astype(float))
    #print(f"Running epoch shape as 2D object: {np.shape(rt)}, with values: {rt}")
    
    starts = rt[:, 0]
    ends   = rt[:, 1]
    
    #print(f"start: {starts}, and end: {ends} of running epoch")
    running_epoch = nap.IntervalSet(start=starts, end=ends)

    pos_full = nap.TsdFrame(
        t=time,
        d=np.c_[x_pos,y_pos],
        columns=["x","y"])
    
    vel_full, dt_full, distance_traveled = calculate_velocity(pos_full)
    #If we do not look at up/down states for the linear track, 
    # we don't really need the movement direction
    movement_dir = calculate_movement_direction(x_pos, y_pos)
    
    #essentially makes arrays for each variable we care about
    hd = nap.Tsd(t=time, d=head_direction)
    vel = nap.Tsd(t=time, d=vel_full)
    md = nap.Tsd(t=time, d=movement_dir)
    dttsd = nap.Tsd(t=time, d=dt_full)
    dist_trav = nap.Tsd(t=time, d=distance_traveled)
    
    dt_full = dttsd.restrict(running_epoch).values
    #Prevents counting large gaps -> only relevant for multiple sessions in same recording
    #Won't change much for single runs
    dt_full[dt_full > 60] = 0
    #restricts dataframe to running epoch only
    pos_r = pos_full.restrict(running_epoch)
    hd_r = hd.restrict(running_epoch).values
    vel_r = vel.restrict(running_epoch).values
    md_r = md.restrict(running_epoch).values
    t_r = pos_r.index
    dist_trav_r = dist_trav.restrict(running_epoch).values
    
    #puts arrays into dataframe
    position_data = nap.TsdFrame(
        t=t_r,
        d=np.c_[pos_r['x'].values,
                pos_r['y'].values,
                vel_r,
                hd_r,         
                md_r,        
                dt_full,
                dist_trav_r],
        columns=['x', 'y', 'velocity', 'head_direction', 'movement_direction', 'time_between_frames', 'distance_traveled'],
        time_support=running_epoch)
    
    spike_file = data_dir / 'Spike_Data.mat'
    
    spike_mat = scipy.io.loadmat(spike_file)
    spike_array = spike_mat['Spike_Data']
    excitatory_neurons = spike_mat['Excitatory_Neurons'].flatten()
    inhibitory_neurons = spike_mat['Inhibitory_Neurons'].flatten()
    # Create spike TsGroup
    spike_dict = {}
    unique_cells = np.unique(spike_array[:, 1]).astype(int)
    #Assigns spikes based on cell id
    for cell_id in unique_cells:
        cell_spikes = spike_array[spike_array[:, 1] == cell_id, 0]
        spike_dict[cell_id] = nap.Ts(t=cell_spikes)
    
    
    spike_data_full = nap.TsGroup(spike_dict)
    spike_rt = spike_data_full.restrict(running_epoch)
    
    #print("Everything loaded correctly, yayyyy!!! :)")
    return position_data, spike_rt, excitatory_neurons, inhibitory_neurons, running_epoch


import os
import struct
from pathlib import Path
import numpy as np
import pynapple as nap
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from neo.io import NeuralynxIO


def parse_ncs_timestamps(csc_file, samples_per_record=512):
    """
    Read per-record timestamps from a Neuralynx .ncs file and interpolate
    to per-sample times. This is the ground truth for sample timing,
    accounts for clock drift and any small inter-record gaps that uniform
    t_start + arange/fs computation would smooth over.

    Returns: 1D array of times in seconds, one per sample.
    """
    record_size = 1044   # 8 (ts) + 4 (chan) + 4 (sf) + 4 (nv) + 512*2 (samples)
    header_size = 16 * 1024

    with open(csc_file, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(header_size)
        body = f.read(file_size - header_size)

    n_records      = len(body) // record_size
    record_starts  = np.zeros(n_records, dtype=np.float64)

    # Parse the first 8 bytes (uint64 timestamp in microseconds) of each record
    for i in range(n_records):
        offset                = i * record_size
        ts_microseconds       = struct.unpack('<Q', body[offset : offset + 8])[0]
        record_starts[i]      = ts_microseconds / 1e6

    # Interpolate timestamps within each record using neighbor spacing.
    # Each record holds 512 samples; the inter-sample dt is record_dt/512.
    all_times = np.empty(n_records * samples_per_record, dtype=np.float64)
    for i in range(n_records - 1):
        record_dt   = (record_starts[i + 1] - record_starts[i]) / samples_per_record
        s           = i * samples_per_record
        all_times[s : s + samples_per_record] = (
            record_starts[i] + np.arange(samples_per_record) * record_dt
        )
    # Last record: use the median inter-record dt
    nominal_dt = np.median(np.diff(record_starts)) / samples_per_record
    s          = (n_records - 1) * samples_per_record
    all_times[s : s + samples_per_record] = (
        record_starts[-1] + np.arange(samples_per_record) * nominal_dt
    )
    return all_times


def load_lfp_data(position_data, spike_data, basepath, animal, experiment, csc_name):
    datapath = Path(basepath) / animal / experiment
    csc_file = datapath / csc_name

    #  Load samples via Neo
    exclude_items = []
    for item in os.listdir(datapath):
        item_path = datapath / item
        if item_path.is_dir() or not item.endswith(('.ncs', '.nev', '.ntt', '.nse', '.nvt')):
            exclude_items.append(item)

    reader = NeuralynxIO(dirname=str(datapath), exclude_filenames=exclude_items)
    block  = reader.read_block(lazy=False, signal_group_mode='split-all')

    analog_signal    = block.segments[0].analogsignals[0]
    Sample_Frequency = float(analog_signal.sampling_rate.magnitude)
    Samples_all      = analog_signal.magnitude.flatten()

    # Get accurate per-sample times from the .ncs file directly 
    Times_all = parse_ncs_timestamps(csc_file)

    # should match number of samples Neo returned
    if len(Times_all) != len(Samples_all):
        print(f"WARNING: timestamp count {len(Times_all)} != sample count {len(Samples_all)}")
        n_min     = min(len(Times_all), len(Samples_all))
        Times_all = Times_all[:n_min]
        Samples_all = Samples_all[:n_min]

    lfp_start, lfp_end = Times_all[0], Times_all[-1]
    data_full = nap.Tsd(t=Times_all, d=Samples_all, time_units='s')

    # Quick drift diagnostic for confidence
    nominal_dt = 1.0 / Sample_Frequency
    actual_dts = np.diff(Times_all)
    drift_ppm  = (np.median(actual_dts) - nominal_dt) / nominal_dt * 1e6
    print(f"Sample rate: nominal={Sample_Frequency:.2f} Hz, "
        f"actual median dt={1/np.median(actual_dts):.4f} Hz, "
        f"drift={drift_ppm:+.1f} ppm")

    #  Identify which sessions have valid overlap 
    run_intervals = position_data.time_support
    n_sessions    = len(run_intervals)

    has_spike_data = len(spike_data) > 0
    spike_session_ranges = []
    if has_spike_data:
        for i in range(n_sessions):
            iset_i   = nap.IntervalSet(start=run_intervals.start[i], end=run_intervals.end[i])
            spikes_i = spike_data.restrict(iset_i)
            times_i  = [spikes_i[c].t for c in spikes_i.keys() if len(spikes_i[c].t) > 0]
            if times_i:
                all_t = np.concatenate(times_i)
                spike_session_ranges.append((all_t.min(), all_t.max()))
            else:
                spike_session_ranges.append((np.nan, np.nan))

    print(f'LFP data range: {lfp_start:.1f} - {lfp_end:.1f}s '
        f'(duration: {lfp_end - lfp_start:.1f}s)')
    print(f'Found {n_sessions} run session(s)')

    sessions_ok = []
    for i in range(n_sessions):
        sess_start, sess_end = run_intervals.start[i], run_intervals.end[i]
        print(f'  Session {i+1}:')
        print(f'    Position: {sess_start:.1f} - {sess_end:.1f}s')

        if has_spike_data:
            sp_start, sp_end = spike_session_ranges[i]
            if not np.isnan(sp_start):
                print(f'    Spikes:   {sp_start:.1f} - {sp_end:.1f}s')
            else:
                print(f'    Spikes:   (none in this session)')

        if sess_end < lfp_start or sess_start > lfp_end:
            print(f'    -> NO LFP overlap, session dropped')
            continue

        actual_start = max(sess_start, lfp_start)
        actual_end   = min(sess_end, lfp_end)

        if has_spike_data and not np.isnan(spike_session_ranges[i][0]):
            sp_start, sp_end = spike_session_ranges[i]
            overlap_start    = max(lfp_start, sess_start, sp_start)
            overlap_end      = min(lfp_end, sess_end, sp_end)
        else:
            overlap_start, overlap_end = actual_start, actual_end

        if overlap_start < overlap_end:
            print(f'    -> overlap {overlap_start:.1f} - {overlap_end:.1f}s')
            sessions_ok.append(i)
        else:
            print(f'    -> no three-way overlap, session dropped')

    # Per-session bandpass + Hilbert 
    Lower_Bound, Upper_Bound = 6, 12   # Hz
    nyquist = Sample_Frequency / 2
    sos     = butter(4, [Lower_Bound / nyquist, Upper_Bound / nyquist],
                    btype='band', output='sos')
    sigma   = Sample_Frequency * 0.3   # 300 ms

    t_parts, raw_parts, filt_parts = [], [], []
    amp_parts, phase_parts, pow_parts = [], [], []
    effective_starts, effective_ends = [], []

    for i in sessions_ok:
        s = max(run_intervals.start[i], lfp_start)
        e = min(run_intervals.end[i], lfp_end)
        effective_starts.append(s)
        effective_ends.append(e)

        seg   = data_full.restrict(nap.IntervalSet(start=s, end=e))
        t_seg = seg.index.values
        x_seg = seg.values

        # Skip empty/too-short segments (filtfilt needs > 3 * filter order * 2 samples)
        if len(x_seg) < 100:
            print(f"  Session {i+1}: segment too short ({len(x_seg)} samples), skipping")
            continue

        filt     = sosfiltfilt(sos, x_seg)
        analytic = hilbert(filt)
        amp      = np.abs(analytic)
        amp_sm   = gaussian_filter1d(amp, sigma)
        phase    = np.angle(analytic)
        power    = amp ** 2

        t_parts.append(t_seg)
        raw_parts.append(x_seg)
        filt_parts.append(filt)
        amp_parts.append(amp_sm)
        phase_parts.append(phase)
        pow_parts.append(power)

    t_run              = np.concatenate(t_parts)
    Samples_run        = np.concatenate(raw_parts)
    Filtered_LFP       = np.concatenate(filt_parts)
    Amplitude_smoothed = np.concatenate(amp_parts)
    Phase              = np.concatenate(phase_parts)
    Power              = np.concatenate(pow_parts)

    effective_support = nap.IntervalSet(
        start=np.array(effective_starts),
        end=np.array(effective_ends),
    )

    lfp_data = {
        'filtered_lfp':  nap.Tsd(t=t_run, d=Filtered_LFP,       time_units='s'),
        'amplitude':     nap.Tsd(t=t_run, d=Amplitude_smoothed, time_units='s'),
        'power':         nap.Tsd(t=t_run, d=Power,              time_units='s'),
        'phase':         nap.Tsd(t=t_run, d=Phase,              time_units='s'),
        'raw_lfp':       nap.Tsd(t=t_run, d=Samples_run,        time_units='s'),
        'sampling_rate': Sample_Frequency,
        'run_interval':  effective_support,
        'metadata': {
            'animal':            animal,
            'experiment':        experiment,
            'csc_file':          csc_name,
            'n_sessions':        len(sessions_ok),
            'session_intervals': list(zip(effective_starts, effective_ends)),
            'lfp_data_range':    (lfp_start, lfp_end),
            'clock_drift_ppm':   drift_ppm,
        },
    }
    return lfp_data

def split_multi_session_data(data, expt_name):
    """
    Split a loaded experiment's data into separate single-session entries.
    
    A single-session experiment is returned as {expt_name: data} unchanged.
    A multi-session experiment is returned as
        {f"{expt_name}_run1": data1, f"{expt_name}_run2": data2, ...}
    
    """
    position_data = data['position']
    spike_data    = data['spikes']
    lfp_data      = data['lfp']
    
    pos_intervals = position_data.time_support
    n_sub = len(pos_intervals)
    
    if n_sub == 1:
        return {expt_name: data}
    
    lfp_intervals = lfp_data['run_interval']
    out = {}
    
    for i in range(n_sub):
        sub_iset  = pos_intervals[i:i+1]
        sub_name  = f"{expt_name}_run{i+1}"
        sub_start = pos_intervals.start[i]
        sub_end   = pos_intervals.end[i]
        
        # Match LFP interval to this position sub-session by overlap
        lfp_match_idx = None
        for j in range(len(lfp_intervals)):
            if lfp_intervals.end[j] >= sub_start and lfp_intervals.start[j] <= sub_end:
                lfp_match_idx = j
                break
        
        if lfp_match_idx is None:
            print(f"  Warning: no LFP overlap for {sub_name}, skipping")
            continue
        
        lfp_sub_iset = lfp_intervals[lfp_match_idx:lfp_match_idx+1]
        
        lfp_sub = {
            'filtered_lfp':  lfp_data['filtered_lfp'].restrict(lfp_sub_iset),
            'amplitude':     lfp_data['amplitude'].restrict(lfp_sub_iset),
            'power':         lfp_data['power'].restrict(lfp_sub_iset),
            'phase':         lfp_data['phase'].restrict(lfp_sub_iset),
            'raw_lfp':       lfp_data['raw_lfp'].restrict(lfp_sub_iset),
            'sampling_rate': lfp_data['sampling_rate'],
            'run_interval':  lfp_sub_iset,
            'metadata': {
                **lfp_data['metadata'],
                'n_sessions':        1,
                'session_intervals': [(lfp_sub_iset.start[0], lfp_sub_iset.end[0])],
                'parent_experiment': expt_name,
                'sub_session_index': i,
            },
        }
        
        out[sub_name] = {
            **data,
            'position':      position_data.restrict(sub_iset),
            'spikes':        spike_data.restrict(sub_iset),
            'lfp':           lfp_sub,
            'running_epoch': sub_iset,
        }
    
    return out

# def load_lfp_data(position_data, spike_data, basepath, animal, experiment, csc_name):
#     datapath = Path(basepath) / animal / experiment
    
#     #this will work even if you have multiple run sessions 
#     run_intervals = position_data.time_support
#     n_sessions = len(run_intervals)
    
#     #Extract spike data from each run session in experiment
#     has_spike_data = len(spike_data) > 0
#     spike_session_ranges = []
#     if has_spike_data:
#         for i in range(n_sessions):
#             iset_i = nap.IntervalSet(
#                 start=run_intervals.start[i],
#                 end=run_intervals.end[i],)
#             spikes_i = spike_data.restrict(iset_i)
#             times_i = [spikes_i[c].t for c in spikes_i.keys() if len(spikes_i[c].t) > 0]
#             if times_i:
#                 all_t = np.concatenate(times_i)
#                 spike_session_ranges.append((all_t.min(), all_t.max()))
#             else:
#                 spike_session_ranges.append((np.nan, np.nan))
    
#     #Load LFP data using Neo, then convert to pynapple
#     csc_file = datapath / csc_name

#     exclude_items = []
#     for item in os.listdir(datapath):
#         item_path = datapath / item
#         if item_path.is_dir() or not item.endswith(('.ncs', '.nev', '.ntt', '.nse', '.nvt')):
#             exclude_items.append(item)

#     reader = NeuralynxIO(dirname=str(datapath), exclude_filenames=exclude_items)
#     block = reader.read_block(lazy=False, signal_group_mode='split-all')
#     analog_signal = block.segments[0].analogsignals[0]

#     Sample_Frequency = float(analog_signal.sampling_rate.magnitude)
#     Samples_all = analog_signal.magnitude.flatten()
    
#     Times_all = analog_signal.times.rescale('s').magnitude

#     with open(csc_file, 'rb') as f:
#         f.seek(16 * 1024)
#         first_timestamp_bytes = f.read(8)
#         first_timestamp_microseconds = struct.unpack('<Q', first_timestamp_bytes)[0]
#         time_offset = first_timestamp_microseconds / 1e6

#     dt = 1.0 / Sample_Frequency
#     #Times_all = time_offset + np.arange(len(Samples_all)) * dt
#     Times_all = analog_signal.times.rescale('s').magnitude
#     lfp_start, lfp_end = Times_all[0], Times_all[-1]
#     data_full = nap.Tsd(t=Times_all, d=Samples_all, time_units='s')
    
#     ##Now check to make sure each session overlaps

#     # Check if run period overlaps with LFP data
#     print(f'LFP data range:  {lfp_start:.1f} - {lfp_end:.1f}s '
#         f'(duration: {lfp_end - lfp_start:.1f}s)')
#     print(f'Found {n_sessions} run session(s)')

#     sessions_ok = []
#     for i in range(n_sessions):
#         sess_start = run_intervals.start[i]
#         sess_end = run_intervals.end[i]

#         print(f'  Session {i+1}:')
#         print(f'    Position: {sess_start:.1f} - {sess_end:.1f}s '
#             f'(duration: {sess_end - sess_start:.1f}s)')

#         if has_spike_data:
#             sp_start, sp_end = spike_session_ranges[i]
#             if not np.isnan(sp_start):
#                 print(f'    Spikes:   {sp_start:.1f} - {sp_end:.1f}s')
#             else:
#                 print(f'    Spikes:   (none in this session)')

#         # LFP overlap with this session
#         if sess_end < lfp_start or sess_start > lfp_end:
#             print(f'-> NO LFP overlap, session will be dropped')
#             continue

#         actual_start = max(sess_start, lfp_start)
#         actual_end = min(sess_end, lfp_end)

#         if has_spike_data and not np.isnan(spike_session_ranges[i][0]):
#             sp_start, sp_end = spike_session_ranges[i]
#             overlap_start = max(lfp_start, sess_start, sp_start)
#             overlap_end = min(lfp_end, sess_end, sp_end)
#         else:
#             overlap_start, overlap_end = actual_start, actual_end

#         if overlap_start < overlap_end:
#             print(f'-> all data overlap in region: '
#                 f'{overlap_start:.1f} - {overlap_end:.1f}s '
#                 f'(extracted: {actual_start:.1f} - {actual_end:.1f}s)')
#             sessions_ok.append(i)
#         else:
#             print(f'-> no three-way overlap, session will be dropped')
        

#     ##Run filter + Hilber per session 
#     Lower_Bound, Upper_Bound = 6, 12  # Hz
#     nyquist = Sample_Frequency / 2
#     sos = butter(4, [Lower_Bound / nyquist, Upper_Bound / nyquist],
#                 btype='band', output='sos')
#     sigma = Sample_Frequency * 0.3  # 300 ms

#     t_parts, raw_parts, filt_parts = [], [], []
#     amp_parts, phase_parts, pow_parts = [], [], []

#     effective_starts, effective_ends = [], []
#     for i in sessions_ok:
#         s = max(run_intervals.start[i], lfp_start)
#         e = min(run_intervals.end[i], lfp_end)
#         effective_starts.append(s)
#         effective_ends.append(e)

#         seg = data_full.restrict(nap.IntervalSet(start=s, end=e))
#         t_seg = seg.index.values
#         x_seg = seg.values


#         filt = sosfiltfilt(sos, x_seg)
#         analytic = hilbert(filt)
#         amp = np.abs(analytic)
#         amp_sm = gaussian_filter1d(amp, sigma)
#         phase = np.angle(analytic)
#         power = amp ** 2

#         t_parts.append(t_seg)
#         raw_parts.append(x_seg)
#         filt_parts.append(filt)
#         amp_parts.append(amp_sm)
#         phase_parts.append(phase)
#         pow_parts.append(power)

#     t_run = np.concatenate(t_parts)
#     Samples_run = np.concatenate(raw_parts)
#     Filtered_LFP = np.concatenate(filt_parts)
#     Amplitude_smoothed = np.concatenate(amp_parts)
#     Phase = np.concatenate(phase_parts)
#     Power = np.concatenate(pow_parts)

#     effective_support = nap.IntervalSet(
#         start=np.array(effective_starts),
#         end=np.array(effective_ends),
#     )
    
    
#     # Create dictionary with pynapple  Tsd (time series) object from the full data
#     lfp_data = {
#         'filtered_lfp': nap.Tsd(t=t_run, d=Filtered_LFP, time_units='s'),
#         'amplitude':    nap.Tsd(t=t_run, d=Amplitude_smoothed, time_units='s'),
#         'power':        nap.Tsd(t=t_run, d=Power, time_units='s'),
#         'phase':        nap.Tsd(t=t_run, d=Phase, time_units='s'),
#         'raw_lfp':      nap.Tsd(t=t_run, d=Samples_run, time_units='s'),
#         'sampling_rate': Sample_Frequency,
#         'run_interval': effective_support,
#         'metadata': {
#             'animal': animal,
#             'experiment': experiment,
#             'csc_file': csc_name,
#             'n_sessions': len(sessions_ok),
#             'session_intervals': list(zip(effective_starts, effective_ends)),
#             'lfp_data_range': (lfp_start, lfp_end),
#         },
#     }
#     return lfp_data

