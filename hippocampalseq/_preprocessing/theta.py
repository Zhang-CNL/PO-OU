import numpy as np
import pynapple as nap
import hippocampalseq.utils as hseu
from typing import Optional

def extract_trajectories(run_position_data, starts, ends):
    true_trajectories = []
    for start,end in zip(starts,ends):
        run_subset = run_position_data.restrict(nap.IntervalSet(start,end))
        trajectory = run_subset[['x','y']].values
        true_trajectories.append(trajectory)
    return true_trajectories

def select_run_snippets(
        run_position_data, 
        velocity_cutoff: float = 5.0,
        run_period_threshold: float = 2.0, 
        duration_scaling_factor: float = 2.9 * 6.75
    ):
    mask = run_position_data['velocity'].values >= velocity_cutoff 
    run_starts,run_ends = hseu.extract_times_from_boolean(mask, run_position_data.index.values)
    lengths = run_ends - run_starts
    periods = lengths > run_period_threshold
    starts,ends = run_starts[periods],run_ends[periods]

    true_trajectories = extract_trajectories(run_position_data, starts, ends)
    return starts, ends, true_trajectories

def process_theta(
        run_position_data,
        run_spikes,
        place_cell_ids,
        velocity_cutoff = 5.0,
        run_period_threshold: float = 2.0,
        place_field_scaling_factor: float = 2.9,
        velocity_scaling_factor: float = 6.75,
        time_window_ms: float = 250.0,
        time_window_advance_ms: Optional[float] = None,
    ):
    time_window_s = time_window_ms / 1000
    if time_window_advance_ms is None:
        time_window_advance_ms = time_window_ms
    time_window_advance_s = time_window_advance_ms / 1000
    duration_scaling_factor = velocity_scaling_factor * place_field_scaling_factor

    (
        starts, 
        ends,
        true_trajectories
    ) = select_run_snippets(
        run_position_data, 
        velocity_cutoff,
        run_period_threshold,
        duration_scaling_factor
    )
    ncells = len(run_spikes)
    spikemats = []
    for start,end in zip(starts,ends):
        bins = np.arange(start, end, time_window_advance_s)
        spikemat = np.zeros((len(bins), ncells), dtype=int)
        epoch = nap.IntervalSet(start, end)
        spikes = run_spikes.restrict(epoch)

        #"""
        if np.isclose(time_window_s, time_window_advance_s):
            spikemat = spikes.count(time_window_s, ep=epoch).values
        else:
            t,uids = [],[]
            for uid, ts in spikes.items():
                times = ts.index.values
                t.append(times)
                uids.append(np.full(len(times), uid))

            t = np.concatenate(t)
            uids = np.concatenate(uids)

            sidx = np.argsort(t)
            t = t[sidx]
            uids = uids[sidx]
            
            start_idx = np.searchsorted(t, bins, side='left')
            end_idx   = np.searchsorted(t, bins + time_window_s, side='right')
            for i in range(len(bins)):
                wuid = uids[start_idx[i]:end_idx[i]]
                if len(wuid) > 0:
                    counts = np.bincount(wuid, minlength=ncells)
                    spikemat[i,:] = counts
        """
        for i,bin in enumerate(bins):
            iv = nap.IntervalSet(bin, bin + time_window_s)
            spike_subset = run_spikes.restrict(iv)
            for uid,ts in spike_subset.items():
                if len(ts) > 0:
                    spikemat[i,uid] = len(ts)
        #"""
            
        spikemats.append(spikemat[:,place_cell_ids].astype(int))

    return true_trajectories,spikemats
            