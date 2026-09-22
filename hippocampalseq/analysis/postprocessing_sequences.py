
import numpy as np
import hippocampalseq.utils as hseu
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

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


def peak_posterior_new(emission_probabilities, environment_size, segments, spikemats, *,
                        bin_size_cm=2, phase_bin=10, minimum_max=0.01, half_width=30,
                        velocity_cutoff=10.0, spike_count_percentile=66.67,
                        count_window=(80, 240), theta_start=70.0,
                        theta_length_s=(0.08, 0.16), step_s=5e-3, direction=None):
    """Per-position peak-phase histogram (Wang et al. 2020).

    Oscillations are kept if every window is above velocity_cutoff, the cycle length is
    within theta_length_s, phase is monotonic within the cycle, and (optionally) the run
    direction matches. Of those, the top (100 - spike_count_percentile)% by spike count
    inside count_window ("firing near the peak of theta") are used.
    """
    grid = hseu.make_ndgrid(environment_size, bin_size_cm, indexing='ij').numpy()[:, 0]
    axis_col = environment_size.axes()[0]
    n_pos = 2 * half_width + 1
    dist_edges = (np.arange(n_pos + 1) - half_width - 0.5) * bin_size_cm
    n_ph = int(np.ceil(360 / phase_bin))
    min_max = minimum_max

    rel, phase_all, osc_all, vel_all, spk_all, dir_all, t_all = [], [], [], [], [], [], []
    osc_offset = 0

    for ep, frame, spk in zip(emission_probabilities, segments, spikemats):
        ep = ep.detach().numpy().reshape(len(ep), -1)
        pos = frame[axis_col].values
        d = np.sign(pos[-1] - pos[0]) or 1.0
        R = np.zeros((n_pos, len(ep)))
        for t in range(len(ep)):
            dbin = np.digitize((grid - pos[t]) * d, dist_edges) - 1
            ok = (dbin >= 0) & (dbin < n_pos)
            np.add.at(R[:, t], dbin[ok], ep[t][ok])
        rel.append(R)
        ph = np.mod(frame['Phase Deg'].values, 360.0); ph[ph == 0] = 360.0
        phase_all.append(ph)
        osc_all.append(frame['Oscillation Number'].values.astype(int) + osc_offset)
        osc_offset = osc_all[-1].max() + 1
        vel_all.append(frame['Velocity'].values)
        spk_all.append(np.asarray(spk).sum(axis=1))
        dir_all.append(np.full(len(ep), d))
        t_all.append(frame.index.values)

    R = np.concatenate(rel, axis=1); phase = np.concatenate(phase_all); osc = np.concatenate(osc_all)
    vel = np.concatenate(vel_all); spk = np.concatenate(spk_all); dirs = np.concatenate(dir_all)
    t_all = np.concatenate(t_all)

    lo, hi = count_window
    def in_window(ph):
        return (ph > lo) & (ph <= hi) if lo < hi else (ph > lo) | (ph <= hi)

    info = {}
    for o in np.unique(osc):
        m = np.flatnonzero((osc == o) & (spk > 0))
        if len(m) < 2 or np.any(vel[m] < velocity_cutoff):
            continue
        if direction is not None and dirs[m[0]] != direction:
            continue
        dur = t_all[m[-1]] - t_all[m[0]] + step_s
        if not (theta_length_s[0] <= dur <= theta_length_s[1]):
            continue
        ph = phase[m].copy(); ph[ph <= theta_start] += 360.0
        if np.any(np.diff(ph) < 0):
            continue
        info[o] = spk[m][in_window(phase[m])].sum()

    counts = np.array(list(info.values()))
    thr = np.percentile(counts, spike_count_percentile) if len(counts) else np.inf
    selected = [o for o, s in info.items() if s >= thr]
    print(f"{len(info)} oscillations pass flags; {len(selected)} selected "
          f"(>= {thr:.0f} spikes in {count_window})")

    dist_of_peak = np.zeros((n_pos, n_ph))
    for o in selected:
        m = (osc == o) & (spk > 0)
        if m.sum() < 2:
            continue
        cur, ph = R[:, m], phase[m]
        mx = cur.max(axis=1)
        for p in range(n_pos):
            if mx[p] >= min_max and not np.all(cur[p] == mx[p]):
                for phv in ph[cur[p] == mx[p]]:
                    dist_of_peak[p, np.clip(int(np.ceil(phv / phase_bin)) - 1, 0, n_ph - 1)] += 1
    col = dist_of_peak.sum(axis=0, keepdims=True); col[col == 0] = 1
    return {"dist_of_peak_post": dist_of_peak, "norm_dist": dist_of_peak / col,
            "n_oscillations_used": len(selected), "spike_count_threshold": thr}

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

def _two_troughs(trace, phase_bin, smooth_sigma=1.0):
    n = len(trace)
    x = np.concatenate([trace, trace, trace])
    if smooth_sigma:
        x = gaussian_filter1d(x, smooth_sigma)
    idx, _ = find_peaks(-x)
    idx = np.unique(idx[(idx >= n) & (idx < 2 * n)] - n)
    j0 = idx[np.argmin(x[idx + n])]  # deepest local minimum
    if len(idx) == 1:
        j1 = (j0 + n // 2) % n
    else:
        other = idx[idx != j0]
        sep = np.abs(((other - j0 + n // 2) % n) - n // 2) * phase_bin
        j1 = other[np.argmin(np.abs(sep - 180.0))]   # the one nearest opposite
    return np.sort([j0, j1]) * phase_bin + phase_bin / 2


def identify_theta_sequence_windows(bimodal_firing_rate, decoding_distribution, phase_bin=10):
    fr_t  = _two_troughs(np.asarray(bimodal_firing_rate), phase_bin, smooth_sigma=1.0)
    dec_t = _two_troughs(np.asarray(decoding_distribution).max(axis=0), phase_bin, smooth_sigma=1.0)
    w = lambda a, b: [a % 360, b % 360]
    return {
        'major_peak_window': w(fr_t[1] + phase_bin,  fr_t[0] - phase_bin),
        'minor_peak_window': w(fr_t[0],              fr_t[1]),
        'forward_window':    w(dec_t[1] + phase_bin, dec_t[0] - phase_bin),
        'reverse_window':    w(dec_t[0] + phase_bin, dec_t[1] - phase_bin),
        'firing_rate_troughs_deg': fr_t.tolist(),
        'decoding_troughs_deg':    dec_t.tolist(),
    }