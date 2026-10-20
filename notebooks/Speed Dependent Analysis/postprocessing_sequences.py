
import numpy as np
import hippocampalseq.utils as hseu
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import warnings
import os
from datetime import datetime

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import hippocampalseq.utils as hseu



def gaussian_to_grid(means, covs, environment_size, bin_size_cm, latent_dim):
    """Turn per-window Gaussian posteriors into (T, Nx, 1) grids like emission_probabilities."""
    Z = hseu.make_ndgrid(environment_size, bin_size_cm, indexing='ij')
    out = []
    for m, c in zip(means, covs):
        m = m[:, latent_dim:].detach().double()
        c = torch.atleast_2d(c[:, latent_dim:, latent_dim:].detach().double())
        g = torch.zeros((len(m), len(Z)), dtype=torch.double)
        for t in range(len(m)):
            mvn = MultivariateNormal(m[t].ravel(), scale_tril=torch.linalg.cholesky(c[t]))
            lp = mvn.log_prob(Z)
            g[t] = torch.exp(lp - torch.logsumexp(lp, 0))
        out.append(g[:, :, None])                      # (T, Nx, 1)
    return out


def peak_posterior(
    decoded_grids,
    environment_size,
    segments,
    spikemats,
    *,
    bin_size_cm=2,
    phase_bin=10,
    half_width=30,
    minimum_max=0.01,
    velocity_cutoff=10.0,
    spike_count_percentile=66.67,
    count_window=None,
    theta_start=70.0,
    theta_length_s=(0.08, 0.16),
    step_s=5e-3,
    direction=None,
    verbose=True,
     velocity_range=None
):
    """Per-position peak-phase histogram (mimicking Wang et al. 2020, Fig. 1B).
 
    For each qualifying theta cycle and each rat-relative position bin, find the
    phase at which that bin attained its maximum posterior within the cycle and
    add one count there. Columns are then normalised independently.
 
    """
    grid = hseu.make_ndgrid(environment_size, bin_size_cm, indexing="ij").numpy()[:, 0]
    # axis_col = environment_size.axes()[0]
        # environment_size.axes()[0] is the DECLARED axis and is wrong in sessions
    # where the track runs along y. The fields are flattened onto the long axis
    # either way, so the grid carries the track's length: pick the column whose
    # travelled span matches it. Had to add this to make sure that there isn't a mismatch between the grid and the position column.
    axis_col = min([c for c in ("x", "y") if c in segments[0].columns],
                key=lambda c: abs(float(np.ptp(np.concatenate(
                    [s[c].values for s in segments]))) - float(np.ptp(grid))))
    n_pos = 2 * half_width + 1
    dist_edges = (np.arange(n_pos + 1) - half_width - 0.5) * bin_size_cm
    n_ph = int(np.ceil(360 / phase_bin))

    rel, phase_all, osc_all, vel_all, spk_all, dir_all, t_all = [], [], [], [], [], [], []
    osc_offset = 0
    for ep, frame, spk in zip(decoded_grids, segments, spikemats):
        ep = np.asarray(ep.detach() if torch.is_tensor(ep) else ep).reshape(len(ep), -1)
        pos = frame[axis_col].values
        d = np.sign(pos[-1] - pos[0]) or 1.0          # one sign per run segment
        R = np.zeros((n_pos, len(ep)))
        for t in range(len(ep)):
            dbin = np.digitize((grid - pos[t]) * d, dist_edges) - 1
            ok = (dbin >= 0) & (dbin < n_pos)
            np.add.at(R[:, t], dbin[ok], ep[t][ok])
        rel.append(R)

        ph = np.mod(frame["Phase Deg"].values, 360.0)
        ph[ph == 0] = 360.0
        phase_all.append(ph)
        osc_all.append(frame["Oscillation Number"].values.astype(int) + osc_offset)
        osc_offset = osc_all[-1].max() + 1
        vel_all.append(frame["Velocity"].values)
        spk_all.append(np.asarray(spk).sum(axis=1))
        dir_all.append(np.full(len(ep), d))
        t_all.append(frame.index.values)

    R = np.concatenate(rel, axis=1)
    phase = np.concatenate(phase_all)
    osc = np.concatenate(osc_all)
    vel = np.concatenate(vel_all)
    spk = np.concatenate(spk_all)
    dirs = np.concatenate(dir_all)
    t_all = np.concatenate(t_all)

    lo, hi = count_window
    in_window = (lambda p: (p > lo) & (p <= hi)) if lo < hi else (lambda p: (p > lo) | (p <= hi))

    # cycle selection: velocity, duration, monotonic phase, then spike count 
    info = {}
    for o in np.unique(osc):
        m = np.flatnonzero(osc == o)
        if len(m) < 2 or np.any(vel[m] < velocity_cutoff):
            continue
        if velocity_range is not None:
            v = float(np.median(vel[m]))
            if not (velocity_range[0] <= v < velocity_range[1]):
                continue
        if direction is not None and dirs[m[0]] != direction:
            continue
        dur = t_all[m[-1]] - t_all[m[0]] + step_s
        if not (theta_length_s[0] <= dur <= theta_length_s[1]):
            continue
        ph = phase[m].copy()
        ph[ph <= theta_start] += 360.0
        if np.any(np.diff(ph) < 0):
            continue
        info[o] = spk[m][in_window(phase[m])].sum()

    counts = np.array(list(info.values()))
    thr = np.percentile(counts, spike_count_percentile) if counts.size else np.inf
    selected = [o for o, s in info.items() if s >= thr]
    if verbose:
        print(f"{len(info)} oscillations pass flags; {len(selected)} selected "
                f"(>= {thr:.0f} spikes in {count_window})")

    dist_of_peak = np.zeros((n_pos, n_ph))
    for o in selected:
        m = (osc == o) & (spk > 0)                 # zero-spike windows excluded here
        if m.sum() < 2:
            continue
        cur, ph = R[:, m], phase[m]
        mx = cur.max(axis=1)
        for p in range(n_pos):
            if mx[p] >= minimum_max and not np.all(cur[p] == mx[p]):
                for phv in ph[cur[p] == mx[p]]:
                    dist_of_peak[p, np.clip(int(np.ceil(phv / phase_bin)) - 1, 0, n_ph - 1)] += 1

    col = dist_of_peak.sum(axis=0, keepdims=True)
    col[col == 0] = 1

    params = dict(bin_size_cm=bin_size_cm, phase_bin=phase_bin, half_width=half_width,
                minimum_max=minimum_max, velocity_cutoff=velocity_cutoff,
                spike_count_percentile=spike_count_percentile, count_window=count_window,
                theta_start=theta_start, theta_length_s=theta_length_s,
                step_s=step_s, direction=direction)
    return {"dist_of_peak_post": dist_of_peak,
            "norm_dist": dist_of_peak / col,
            "n_oscillations_used": len(selected),
            "spike_count_threshold": thr,
            "params": params}
 
def save_all_figures(figures_dict, save_dir, filename_tag):
    """Save all figures to organized PDFs."""
    date_folder = datetime.now().strftime('%Y-%m-%d')
    top_folder = os.path.join(save_dir, date_folder, filename_tag)
    os.makedirs(top_folder, exist_ok=True)

    for category, figs in figures_dict.items():
        pdf_path = os.path.join(top_folder, f"{category}.pdf")

        with PdfPages(pdf_path) as pdf:
            if isinstance(figs, dict):
                for name, fig in figs.items():
                    if isinstance(fig, list):
                        for f in fig:
                            pdf.savefig(f, bbox_inches='tight')
                            plt.close(f)
                    else:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            elif isinstance(figs, list):
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            else:
                pdf.savefig(figs, bbox_inches='tight')
                plt.close(figs)
        print(f"Saved {category} to {pdf_path}")

    return top_folder
 
def _two_troughs(trace, phase_bin, smooth_sigma=None, n_tile=2, min_prominence_frac=0.1):
    """First two troughs in phase order (as the MATLAB code does), but requiring each
    to have prominence >= min_prominence_frac of the trace range.

    The MATLAB calls findpeaks with no prominence filter, which is safe only on a
    trace with exactly two minima per cycle. I think Brad perhaps uses something else to exclude noisy neurons.
    On a noisier trace an unfiltered findpeaks returns shallow wiggles on the flank of the real trough, and
    taking idx[:2] then picks a wiggle. I think this is part of the reason I have unimodal and bimodal cells bleeding together
    """
    n = len(trace)
    x = np.tile(np.asarray(trace, float), n_tile)
    if smooth_sigma:
        x = gaussian_filter1d(x, smooth_sigma, mode="wrap")
    prom = min_prominence_frac * (x.max() - x.min())
    idx, props = find_peaks(-x, prominence=prom)
    if len(idx) < 2:
        raise ValueError(f"found {len(idx)} troughs at prominence >= {prom:.4g}; "
                         "lower min_prominence_frac or check the trace")
    first_two = np.sort(idx[:2])
    span = (first_two[1] - first_two[0]) * phase_bin
    if span >= 355:
        raise ValueError(f"only one trough per cycle (span {span:.0f} deg)")
    return first_two * phase_bin + phase_bin / 2
 
def identify_theta_sequence_windows(decoding_distribution, *, minor_window, major_window,
                                    phase_bin=10, smooth_sigma=np.sqrt(2)):
    """Forward/reverse windows from the decoding trace. Major/minor are passed in.

    The pooled bimodal firing rate has only one trough per cycle in this dataset
    (the minor peak is resolved in 5 of 9 sessions individually but washes out
    when pooled), so major/minor are set by hand from the per-session peaks.
    """
    trace = np.asarray(decoding_distribution).max(axis=0)
    sm = gaussian_filter1d(trace, smooth_sigma, mode="wrap")
    dec_t = _two_troughs(sm, phase_bin, smooth_sigma=None)
    w = lambda a, b: [a % 360, b % 360]
    return {
        "major_peak_window": list(major_window),
        "minor_peak_window": list(minor_window),
        "forward_window": w(dec_t[1] + phase_bin, dec_t[0] - phase_bin),
        "reverse_window": w(dec_t[0] + phase_bin, dec_t[1] - phase_bin),
        "decoding_troughs_deg": dec_t.tolist(),
        "max_prob_trace": sm,
    }
    
    
def weighted_fit(dist, col_idx, phase_bin, position_bin, mid):
    """Probability-weighted least squares. Returns (slope cm/deg, intercept cm, weighted r).
 
    The paper's Fig 1B caption says "least-squares", which the MATLAB implements as
    per-column argmax + polyfit. I use that for the faithful reproduction and this as a
    robustness check; the two disagree when two bands coexist in a window. I haven't fully figured
    out what to do about that. I think the least-squares is better actually
    """
    W = dist[:, col_idx].astype(float)
    W = W / W.sum()
    x = (np.asarray(col_idx) * phase_bin + phase_bin / 2)[None, :]
    y = ((np.arange(dist.shape[0]) - mid) * position_bin)[:, None]
    mx, my = (W * x).sum(), (W * y).sum()
    cxx = (W * (x - mx) ** 2).sum()
    cyy = (W * (y - my) ** 2).sum()
    cxy = (W * (x - mx) * (y - my)).sum()
    slope = cxy / cxx
    return slope, my - slope * mx, cxy / np.sqrt(cxx * cyy)
 
 
# Turning decoder output into the grids peak_posterior expects
 
def moments_to_grid(means, covs, environment_size, bin_size_cm, jitter=1e-6):
    """Expand per-window Gaussian moments back onto the position grid.
 
    Works for GaussianApproximation output and for the position block of an SSM's
    smoothed state. Jitter is required: a posterior collapsed onto one or two bins
    gives a near-singular covariance and fails without it.
    """
    Z = hseu.make_ndgrid(environment_size, bin_size_cm, indexing="ij").double()
    D = np.asarray(covs[0]).shape[-1]
    eye = torch.eye(D, dtype=torch.double) * jitter
    out = []
    for m, c in zip(means, covs):
        m = torch.as_tensor(np.asarray(m.detach() if torch.is_tensor(m) else m), dtype=torch.double)
        c = torch.as_tensor(np.asarray(c.detach() if torch.is_tensor(c) else c), dtype=torch.double) + eye
        g = torch.zeros((len(m), len(Z)), dtype=torch.double)
        for t in range(len(m)):
            mvn = MultivariateNormal(m[t].ravel(), scale_tril=torch.linalg.cholesky(c[t]))
            lp = mvn.log_prob(Z)
            g[t] = torch.exp(lp - torch.logsumexp(lp, 0))
        out.append(g[:, :, None])
    return out
 
 
def grids_map(emissions):
    """One-hot per window at the argmax, so MAP runs through the same statistic.
 
    Note: do NOT use BayesianMAP.fit for this — it drops zero-spike windows from the
    spikemat only, so its output no longer aligns with ground_truth. The MATLAB drops
    them too, but trims the time info in lockstep.
    """
    out = []
    for e in emissions:
        e = e.detach() if torch.is_tensor(e) else torch.as_tensor(e)
        flat = e.reshape(len(e), -1)
        g = torch.zeros_like(flat)
        g[torch.arange(len(e)), flat.argmax(1)] = 1.0
        out.append(g.reshape(e.shape))
    return out
 
def session_emissions(place_fields, spikemats, dt):
    """Naive-Bayes posterior per window, float32 for storage."""
    pf = hseu.ensure_torch(place_fields)
    out = []
    for spk in spikemats:
        e = hseu.calc_poisson_emission_probabilities_2d(
            hseu.ensure_torch(spk).double(), pf, torch.tensor(dt))
        e = e / e.sum(dim=(1, 2), keepdim=True)
        out.append(torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0).float())
    return out

def grids_gaussian(emissions, environment_size, bin_size_cm, mode="weighted", jitter=1e-6):
    """Gaussian moment-match of each window's posterior, re-expanded on the grid.

    mode='weighted' matches Momentum.spike_to_gaussian; the GaussianApproximation
    class defaults to 'max', which is MAP-centred and not comparable.
    """
    Z = hseu.make_ndgrid(environment_size, bin_size_cm, indexing="ij").double()
    out = []
    for e in emissions:
        e = e.double() if torch.is_tensor(e) else torch.as_tensor(e).double()
        mu, cov = hseu.analytical_gaussian_approximation(Z, e, mode)
        cov = cov.double() + torch.eye(cov.shape[-1], dtype=torch.double) * jitter
        g = torch.zeros((len(mu), len(Z)), dtype=torch.double)
        for t in range(len(mu)):
            mvn = MultivariateNormal(mu[t].ravel().double(), scale_tril=torch.linalg.cholesky(cov[t]))
            lp = mvn.log_prob(Z)
            g[t] = torch.exp(lp - torch.logsumexp(lp, 0))
        out.append(g.reshape(e.shape))
    return out


def windows_from_firing_rate(bimodal_firing_rate, phase_bin=10):
    """Major/minor windows from the bimodal population rate alone.

    Needed before any decoding, because the paper selects the top third of cycles
    by spike count in the MINOR window which is the reverse window.  Otherwise, the reverse
    sweeps get washed out
    """
    fr_t = _two_troughs(np.asarray(bimodal_firing_rate), phase_bin, smooth_sigma=1.0)
    w = lambda a, b: [a % 360, b % 360]
    return {"major_peak_window": w(fr_t[1] + phase_bin, fr_t[0] - phase_bin),
            "minor_peak_window": w(fr_t[0], fr_t[1]),
            "firing_rate_troughs_deg": fr_t.tolist()}
    
def long_axis(environment_size, bin_size_cm, frames, candidates=("x", "y")):
    """Which position column is the track's long axis for this session.

    environment_size.axes()[0] is the declared axis and is wrong in sessions
    where the track runs along y; the place fields are flattened onto the long
    axis either way, so the grid carries the track's length and the correct
    column is the one whose travelled span matches it.
    """
    lo, hi = environment_size.extent()[0] if hasattr(environment_size, "extent") else (None, None)
    n = int(round((hi - lo) / bin_size_cm)) if lo is not None else None
    grid_span = (hi - lo) if lo is not None else None
    if grid_span is None:                       # fall back: use the posterior width
        raise ValueError("pass the grid span explicitly")
    return min([c for c in candidates if c in frames[0].columns],
            key=lambda c: abs(float(np.ptp(np.concatenate(
                [f[c].values for f in frames]))) - grid_span))