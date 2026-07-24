import os
import struct 
import warnings
import numpy as np
import pynapple as nap
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from neo.io import NeuralynxIO
from typing import Any



def parse_ncs_timestamps(csc_file: str, samples_per_record: int=512) -> np.ndarray:
    """
    Read per-record timestamps from a Neuralynx .ncs file and interpolate
    to per-sample times. This is the ground truth for sample timing,
    accounts for clock drift and any small inter-record gaps that uniform
    t_start + arange/fs computation would smooth over.
    Args:
        csc_file (str): Path to .ncs file.
        samples_per_record (int): Number of samples per record.

    Returns: 
        (np.array): 1D array of times in seconds, one per sample
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

def load_lfp_data(
        position_data: nap.TsdFrame, 
        spike_data: nap.TsGroup, 
        datapath: str
    ) -> dict[str, Any]:
    """Load LFP data from a .ncs file
    Args:
        position_data (nap.TsdFrame): Raw position data from the rat.
        spike_data (nap.TsGroup): Raw spike data.
        datapath (str): Base path to the directory where the .ncs file is stored.
    
    Returns:
        dict: A dict containing LFP data such as amplitude and phase, as well as metadata.
    """

    #  Load samples via Neo
    includes = []
    for item in os.listdir(datapath):
        if item.endswith(('.ncs', '.nev', '.ntt', '.nse', '.nvt')):
            includes.append(item)

    reader = NeuralynxIO(dirname=str(datapath), include_filenames=includes)
    block  = reader.read_block(lazy=False, signal_group_mode='split-all')

    Samples_all = []
    Sample_Frequency = []
    for seg in block.segments:
        for sig in seg.analogsignals:
            Sample_Frequency.append(float(sig.sampling_rate.magnitude))
            Samples_all.append(sig.magnitude.flatten())
            
    Sample_Frequency = float(np.unique(Sample_Frequency)[0])
    Samples_all = np.concatenate(Samples_all)
#    analog_signal    = block.segments[0].analogsignals[0]
#    Sample_Frequency = float(analog_signal.sampling_rate.magnitude)
#    Samples_all      = analog_signal.magnitude.flatten()

    # Get accurate per-sample times from the .ncs file directly 
    Times_all = parse_ncs_timestamps(os.path.join(datapath,includes[0]))

    # should match number of samples Neo returned
    if len(Times_all) != len(Samples_all):
        warnings.warn(f"WARNING: timestamp count {len(Times_all)} != sample count {len(Samples_all)}")
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
                warnings.warn(f'    Spikes:   (none in this session)')

        if sess_end < lfp_start or sess_start > lfp_end:
            warnings.warn(f'    -> NO LFP overlap, session {i+1} dropped')
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
            warnings.warn(f'    -> overlap {overlap_start:.1f} - {overlap_end:.1f}s')
            sessions_ok.append(i)
        else:
            warnings.warn(f'    -> no three-way overlap, session dropped')

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
            warnings.warn(f"  Session {i+1}: segment too short ({len(x_seg)} samples), skipping")
            continue

        filt     = sosfiltfilt(sos, x_seg)
        analytic = hilbert(filt)
        amp      = np.abs(analytic)
        amp_sm   = gaussian_filter1d(amp, sigma)
        phase    = np.angle(analytic)
        power    = amp**2

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

    lfp_data = nap.TsdFrame(
        t=t_run,
        d=np.c_[
            Filtered_LFP,
            Amplitude_smoothed,
            Power,
            Phase,
            Samples_run
        ],
        columns=[
            'Filtered LFP', 'Amplitude', 'Power', 
            'Phase Rad', 'Raw LFP'
        ],
        time_units='s'
    )
    out = {
        'LFP': lfp_data, 
        'sampling_rate': Sample_Frequency, 
        'run_interval' : effective_support,
        'metadata': {
            'n_session': len(sessions_ok),
            'lfp_data_range': (lfp_start, lfp_end),
            'clock_drift_ppm': drift_ppm
        }
    }

    return out