from scipy.signal import filtfilt
from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from pathlib import Path

def plot_phase_locked(firing_rate_per_phase, phase_centers, n_cells=12):
    ids = list(firing_rate_per_phase.keys())[:n_cells]
    n = len(ids)
    if n == 0:
        return None

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes = axes.ravel()

    # Doubled axis for 0-720° display (two cycles of theta)
    phase_doubled = np.concatenate([phase_centers, phase_centers + 360])

    for i, cid in enumerate(ids):
        ax = axes[i]
        d = firing_rate_per_phase[cid]

        raw_d    = np.concatenate([d['raw_counts'],    d['raw_counts']])
        smooth_d = np.concatenate([d['smooth_counts'], d['smooth_counts']])

        ax.bar(phase_doubled, raw_d,
            width=phase_centers[1] - phase_centers[0],
            color='black', edgecolor='none', alpha=0.7, label='Raw')
        ax.plot(phase_doubled, smooth_d, color='red', lw=1.5, label='Smoothed')

        ax.set_title(f"Cell {cid} (n={d['n_spikes']})", fontsize=10)
        ax.set_xlim(0, 720)
        ax.set_xticks([0, 360, 720])
        ax.set_xlabel("Theta phase (°)")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, rows * cols):
        axes[j].axis("off")

    fig.tight_layout()
    return fig

def plot_all_phase_locked(firing_rate_per_phase, phase_centers,
                        cols=4, rows_per_page=5):
    """Plot every cell, paginated. Returns list of figures."""
    ids = list(firing_rate_per_phase.keys())
    n_total = len(ids)
    if n_total == 0:
        return []

    cells_per_page = cols * rows_per_page
    n_pages = int(np.ceil(n_total / cells_per_page))

    phase_doubled = np.concatenate([phase_centers, phase_centers + 360])
    bar_width = phase_centers[1] - phase_centers[0]

    figures = []
    for page_idx in range(n_pages):
        page_ids = ids[page_idx * cells_per_page : (page_idx + 1) * cells_per_page]
        n_on_page = len(page_ids)

        fig, axes = plt.subplots(rows_per_page, cols,
                                figsize=(cols * 2.6, rows_per_page * 2.0),
                                squeeze=False)
        axes = axes.ravel()

        for i, cid in enumerate(page_ids):
            ax = axes[i]
            d = firing_rate_per_phase[cid]

            raw_d    = np.concatenate([d['raw_counts'],    d['raw_counts']])
            smooth_d = np.concatenate([d['smooth_counts'], d['smooth_counts']])

            ax.bar(phase_doubled, raw_d, width=bar_width,
                color='black', edgecolor='none', alpha=0.7)
            ax.plot(phase_doubled, smooth_d, color='red', lw=1.0)

            ax.set_title(f"Cell {cid} (n={d['n_spikes']})", fontsize=7)
            ax.set_xlim(0, 720)
            ax.set_xticks([0, 360, 720])
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=6)

            # Only label left column ylabel and bottom row xlabel
            row, col = divmod(i, cols)
            is_last_row = (i >= n_on_page - cols)
            if col == 0:
                ax.set_ylabel("Count", fontsize=7)
            if is_last_row:
                ax.set_xlabel("Phase (°)", fontsize=7)

        for j in range(n_on_page, cells_per_page):
            axes[j].axis("off")

        fig.suptitle(f"Page {page_idx + 1} / {n_pages}  ({n_total} cells total)",
                    fontsize=9, y=0.995)
        fig.tight_layout()
        figures.append(fig)

    return figures

def plot_modality_classification(firing_rate_per_phase,
                                modality_results, population_stats,
                                phase_centers, n_examples=3):

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    phase_doubled = np.concatenate([phase_centers, phase_centers + 360])
    bar_width     = phase_centers[1] - phase_centers[0]

    # --- Row 1: population FRI averages ---
    groups = ['all_excitatory', 'unimodal', 'bimodal']
    titles = ['All Excitatory', 'Unimodal',  'Bimodal']
    colors = ['gray',           'blue',      'red']

    for i, (group, title, color) in enumerate(zip(groups, titles, colors)):
        ax    = fig.add_subplot(gs[0, i])
        stats = population_stats[group]

        if stats['n_cells'] > 0:
            mean, sem = stats['rate_index']
            mean_d    = np.concatenate([mean, mean])
            if sem is not None:
                sem_d = np.concatenate([sem, sem])
                ax.fill_between(phase_doubled, mean_d - sem_d, mean_d + sem_d,
                                alpha=0.3, color=color)
            ax.plot(phase_doubled, mean_d, color=color, linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No cells', ha='center', va='center',
                    transform=ax.transAxes)

        ax.set_xlabel('Theta Phase (°)')
        ax.set_ylabel('Firing Rate Index')
        ax.set_title(f'{title} (n={stats["n_cells"]})', fontweight='bold')
        ax.set_xlim(0, 720)
        ax.set_xticks([0, 360, 720])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # --- Row 2: example unimodal cells ---
    unimodal_ids = [cid for cid, r in modality_results.items()
                    if r['modality'] == 1][:n_examples]

    for i, cell_id in enumerate(unimodal_ids):
        ax     = fig.add_subplot(gs[1, i])
        data   = firing_rate_per_phase[cell_id]
        result = modality_results[cell_id]

        raw_d    = np.concatenate([data['raw_counts'],    data['raw_counts']])
        smooth_d = np.concatenate([data['smooth_counts'], data['smooth_counts']])

        ax.bar(phase_doubled, raw_d, width=bar_width,
               color='black', edgecolor='none', alpha=0.7)
        ax.plot(phase_doubled, smooth_d, color='blue', lw=1.5)

        for peak_phase in result['peak_phases']:
            ax.axvline(peak_phase,       color='red', linestyle='--', alpha=0.7)
            ax.axvline(peak_phase + 360, color='red', linestyle='--', alpha=0.7)

        ax.set_xlabel('Theta Phase (°)')
        ax.set_ylabel('Count')
        ax.set_title(f"Unimodal Cell {cell_id}\n(p={result['rayleigh_p']:.3f})",
                     fontweight='bold')
        ax.set_xlim(0, 720)
        ax.set_xticks([0, 360, 720])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # --- Row 3: example bimodal cells ---
    bimodal_ids = [cid for cid, r in modality_results.items()
                   if r['modality'] == 2][:n_examples]

    for i, cell_id in enumerate(bimodal_ids):
        ax     = fig.add_subplot(gs[2, i])
        data   = firing_rate_per_phase[cell_id]
        result = modality_results[cell_id]

        raw_d    = np.concatenate([data['raw_counts'],    data['raw_counts']])
        smooth_d = np.concatenate([data['smooth_counts'], data['smooth_counts']])

        ax.bar(phase_doubled, raw_d, width=bar_width,
               color='black', edgecolor='none', alpha=0.7)
        ax.plot(phase_doubled, smooth_d, color='red', lw=1.5)

        for peak_phase in result['peak_phases']:
            ax.axvline(peak_phase,       color='blue', linestyle='--', alpha=0.7)
            ax.axvline(peak_phase + 360, color='blue', linestyle='--', alpha=0.7)

        ax.set_xlabel('Theta Phase (°)')
        ax.set_ylabel('Count')
        ax.set_title(f"Bimodal Cell {cell_id}\n(p={result['rayleigh_p']:.3f})",
                     fontweight='bold')
        ax.set_xlim(0, 720)
        ax.set_xticks([0, 360, 720])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide empty example panels
    for i in range(len(unimodal_ids), n_examples):
        fig.add_subplot(gs[1, i]).axis('off')
    for i in range(len(bimodal_ids), n_examples):
        fig.add_subplot(gs[2, i]).axis('off')

    plt.tight_layout()
    return fig

def plot_modality_all_cells(firing_rate_per_phase, modality_results, phase_centers,
                             cols=4, rows_per_page=5):
    """Plot every cell grouped by modality. Returns list of figures."""

    modality_groups = [
        (1,  'Unimodal',       'blue'),
        (2,  'Bimodal',        'red'),
        (3,  'Multimodal',     'purple'),
        (-1, 'Non-modal',      'gray'),
        (0,  'Too few spikes', 'lightgray'),
    ]

    cells_per_page = cols * rows_per_page
    phase_doubled  = np.concatenate([phase_centers, phase_centers + 360])
    bar_width      = phase_centers[1] - phase_centers[0]

    figures = []

    for mod_val, name, color in modality_groups:
        cell_ids = [cid for cid, r in modality_results.items() if r['modality'] == mod_val]
        if not cell_ids:
            continue

        n_total = len(cell_ids)
        n_pages = int(np.ceil(n_total / cells_per_page))

        for page_idx in range(n_pages):
            page_ids  = cell_ids[page_idx * cells_per_page : (page_idx + 1) * cells_per_page]
            n_on_page = len(page_ids)

            fig, axes = plt.subplots(rows_per_page, cols,
                                      figsize=(cols * 2.6, rows_per_page * 2.0),
                                      squeeze=False)
            axes = axes.ravel()

            for i, cid in enumerate(page_ids):
                ax = axes[i]
                d  = firing_rate_per_phase[cid]
                r  = modality_results[cid]

                raw_d    = np.concatenate([d['raw_counts'],    d['raw_counts']])
                smooth_d = np.concatenate([d['smooth_counts'], d['smooth_counts']])

                ax.bar(phase_doubled, raw_d, width=bar_width,
                       color='black', edgecolor='none', alpha=0.7)
                ax.plot(phase_doubled, smooth_d, color=color, lw=1.0)

                # Mark detected peaks in both cycles
                for pphase in r['peak_phases']:
                    ax.axvline(pphase,       color='red', linestyle='--', alpha=0.5, lw=0.6)
                    ax.axvline(pphase + 360, color='red', linestyle='--', alpha=0.5, lw=0.6)

                ax.set_title(f"Cell {cid}  n={d['n_spikes']}  p={r['rayleigh_p']:.3f}",
                             fontsize=6)
                ax.set_xlim(0, 720)
                ax.set_xticks([0, 360, 720])
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=6)

                row, col_pos = divmod(i, cols)
                if col_pos == 0:
                    ax.set_ylabel("Count", fontsize=7)
                if i >= n_on_page - cols:
                    ax.set_xlabel("Phase (°)", fontsize=7)

            for j in range(n_on_page, cells_per_page):
                axes[j].axis("off")

            fig.suptitle(f"{name} — page {page_idx + 1}/{n_pages}  ({n_total} cells)",
                         fontsize=10, y=0.995, color=color)
            fig.tight_layout()
            figures.append(fig)

    return figures

def plot_place_field_comparison(unimodal_props, bimodal_props, expt_name=None):
    """Compare place field properties between unimodal and bimodal cells."""
    from scipy.stats import mannwhitneyu

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    metrics = [
        ('mean_field_size',         'Mean Place Field Size (bins)'),
        ('n_fields',                'Number of Place Fields'),
        ('peak_firing_rate',        'Peak Firing Rate (Hz)'),
        ('mean_firing_rate',        'Mean Firing Rate (Hz)'),
        ('information_per_spike',   'Information per Spike (bits)'),
        ('mean_infield_firing_rate','Mean In-Field Firing Rate (Hz)'),
    ]

    for ax, (key, label) in zip(axes.ravel(), metrics):
        uni_data = np.asarray(unimodal_props[key], dtype=float)
        bi_data  = np.asarray(bimodal_props[key],  dtype=float)

        # Strip NaNs (cells with no contiguous field)
        uni_data = uni_data[~np.isnan(uni_data)]
        bi_data  = bi_data[~np.isnan(bi_data)]

        if len(uni_data) == 0 and len(bi_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=10)
            continue

        # Box plot
        bp = ax.boxplot([uni_data, bi_data], positions=[1, 2], widths=0.6,
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('blue'); bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red');  bp['boxes'][1].set_alpha(0.5)

        # Jittered scatter
        if len(uni_data) > 0:
            ax.scatter(1 + np.random.randn(len(uni_data)) * 0.05,
                        uni_data, color='blue', alpha=0.5, s=20)
        if len(bi_data) > 0:
            ax.scatter(2 + np.random.randn(len(bi_data)) * 0.05,
                        bi_data, color='red', alpha=0.5, s=20)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Unimodal\n(n={len(uni_data)})',
                             f'Bimodal\n(n={len(bi_data)})'])
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mann-Whitney U test
        if len(uni_data) > 0 and len(bi_data) > 0:
            try:
                stat, p = mannwhitneyu(uni_data, bi_data, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.set_title(f'p={p:.4f} ({sig})', fontsize=10)
            except Exception:
                ax.set_title('', fontsize=10)

    title = 'Unimodal vs Bimodal Place Field Properties'
    if expt_name:
        title += f' — {expt_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_unimodal_bimodal_summary(firing_rate_per_phase, modality_results, population_stats,
                                phase_centers, unimodal_props, bimodal_props,
                                excitatory_neurons, save_dir=None):
    """
    Comprehensive plotting of unimodal vs bimodal cell properties.
    Matches MATLAB: IRFS_PLOT_UNIMODAL_BIMODAL_CELL_PROPERTIES (without waveform parts)
    """
    
    # --- Figure 1: Population phase-locked firing ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    
    groups = ['all_excitatory', 'unimodal', 'bimodal']
    titles = ['All Excitatory', 'Unimodal', 'Bimodal']
    colors = ['black', 'blue', 'red']
    
    # Duplicate phase for 0-720 display (matching MATLAB)
    phase_720 = np.concatenate([phase_centers, phase_centers + 360])
    
    for ax, group, title, color in zip(axes1, groups, titles, colors):
        stats = population_stats.get(group, {'n_cells': 0})
        if stats['n_cells'] > 0 and stats['rate_index'][0] is not None:
            mean, sem = stats['rate_index']
            mean_720 = np.concatenate([mean, mean])
            sem_720 = np.concatenate([sem, sem])
            
            ax.fill_between(phase_720, mean_720 - sem_720, mean_720 + sem_720, 
                           alpha=0.3, color=color)
            ax.plot(phase_720, mean_720, color=color, linewidth=2)
        
        ax.set_xlabel('Theta Phase (°)')
        ax.set_ylabel('Firing Rate Index')
        ax.set_title(f'{title} (n={stats["n_cells"]})')
        ax.set_xlim(0, 720)
        ax.set_ylim(0, 1)
        ax.axvline(360, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    fig1.suptitle('Population Phase-Locked Firing', fontweight='bold')
    fig1.tight_layout()
    
    # --- Figure 2: Overlay comparison ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    for group, color, label in zip(['all_excitatory', 'unimodal', 'bimodal'],
                                    ['black', 'blue', 'red'],
                                    ['All Excitatory', 'Unimodal', 'Bimodal']):
        stats = population_stats.get(group, {'n_cells': 0})
        if stats['n_cells'] > 0 and stats['rate_index'][0] is not None:
            mean, sem = stats['rate_index']
            mean_720 = np.concatenate([mean, mean])
            sem_720 = np.concatenate([sem, sem])
            
            ax2.fill_between(phase_720, mean_720 - sem_720, mean_720 + sem_720, 
                            alpha=0.2, color=color)
            ax2.plot(phase_720, mean_720, color=color, linewidth=2, 
                    label=f'{label} (n={stats["n_cells"]})')
    
    ax2.set_xlabel('Theta Phase (°)')
    ax2.set_ylabel('Firing Rate Index')
    ax2.set_title('Unimodal vs Bimodal Phase-Locked Firing', fontweight='bold')
    ax2.set_xlim(0, 720)
    ax2.axvline(360, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    
    # --- Figure 3: Modality pie chart ---
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    
    # Count modalities for excitatory neurons only
    modality_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    for cell_id, result in modality_results.items():
        if cell_id in excitatory_neurons:
            modality_counts[result['modality']] += 1
    
    # Only plot unimodal, bimodal, multimodal (skip non-modal and low-spike)
    labels = ['Unimodal', 'Bimodal', 'Multimodal']
    sizes = [modality_counts[1], modality_counts[2], modality_counts[3]]
    colors_pie = ['blue', 'red', 'gray']
    
    # Filter out zero counts
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels, sizes, colors_pie = zip(*non_zero)
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                            autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Modality Distribution\n(Unimodal={modality_counts[1]}, '
                     f'Bimodal={modality_counts[2]}, Multimodal={modality_counts[3]})',
                     fontweight='bold')
    fig3.tight_layout()
    
    # --- Figure 4: Place field property comparisons (box plots) ---
    fig4, axes4 = plt.subplots(2, 3, figsize=(14, 9))
    
    metrics = [
        ('mean_field_size', 'Mean Place Field Size (bins)'),
        ('n_fields', 'Number of Place Fields'),
        ('peak_firing_rate', 'Peak Firing Rate (Hz)'),
        ('mean_firing_rate', 'Mean Firing Rate (Hz)'),
        ('information_per_spike', 'Information per Spike (bits)'),
        ('mean_infield_firing_rate', 'Mean In-Field Firing Rate (Hz)'),
    ]
    
    for ax, (key, label) in zip(axes4.ravel(), metrics):
        uni_data = unimodal_props[key]
        bi_data = bimodal_props[key]
        
        # Box plot
        bp = ax.boxplot([uni_data, bi_data], positions=[1, 2], widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', 
                                      markeredgecolor='black', markersize=8))
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)
        
        # Scatter individual points with jitter
        if len(uni_data) > 0:
            jitter = np.random.uniform(-0.15, 0.15, len(uni_data))
            ax.scatter(np.ones(len(uni_data)) + jitter, uni_data, 
                      color='blue', alpha=0.6, s=25, edgecolor='white', linewidth=0.5)
        if len(bi_data) > 0:
            jitter = np.random.uniform(-0.15, 0.15, len(bi_data))
            ax.scatter(np.ones(len(bi_data)) * 2 + jitter, bi_data, 
                      color='red', alpha=0.6, s=25, edgecolor='white', linewidth=0.5)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Unimodal', 'Bimodal'])
        ax.set_ylabel(label)
        ax.set_xlim(0.4, 2.6)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Stats
        if len(uni_data) > 0 and len(bi_data) > 0:
            from scipy.stats import mannwhitneyu
            try:
                stat, p = mannwhitneyu(uni_data, bi_data, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.set_title(f'p={p:.3f} ({sig})', fontsize=10)
            except:
                pass
    
    fig4.suptitle('Unimodal vs Bimodal Place Field Properties', fontsize=14, fontweight='bold')
    fig4.tight_layout()
    
    # Save figures if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        fig1.savefig(save_dir / 'population_phase_locked_firing.png', dpi=150, bbox_inches='tight')
        fig2.savefig(save_dir / 'phase_locked_firing_overlay.png', dpi=150, bbox_inches='tight')
        fig3.savefig(save_dir / 'modality_pie_chart.png', dpi=150, bbox_inches='tight')
        fig4.savefig(save_dir / 'place_field_comparison.png', dpi=150, bbox_inches='tight')
        print(f'Figures saved to {save_dir}')
    
    return fig1, fig2, fig3, fig4

def plot_modality_overlay(population_stats, phase_centers):
    """Three-panel overlay of All/Unimodal/Bimodal populations across phase."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    phase_doubled = np.concatenate([phase_centers, phase_centers + 360])

    groups = [
        ('all_excitatory', 'black', 'All Excitatory'),
        ('unimodal',       'red',   'Unimodal'),
        ('bimodal',        'blue',  'Bimodal'),
    ]

    metrics = [
        ('raw_rate',    'Raw Firing Rate'),
        ('smooth_rate', 'Smoothed Firing Rate'),
        ('rate_index',  'Firing Rate Index'),
    ]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for group_key, color, label in groups:
            stats = population_stats[group_key]
            if stats['n_cells'] == 0:
                continue
            mean, sem = stats[metric_key]
            if mean is None:
                continue
            mean_d = np.concatenate([mean, mean])
            ax.plot(phase_doubled, mean_d, color=color, lw=1.5,
                    label=f"{label} (n={stats['n_cells']})")
            if sem is not None:
                sem_d = np.concatenate([sem, sem])
                ax.fill_between(phase_doubled, mean_d - sem_d, mean_d + sem_d,
                                alpha=0.2, color=color)

        ax.set_xlim(0, 720); ax.set_xticks([0, 360, 720])
        ax.set_xlabel('Theta Phase (°)')
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    fig.tight_layout()
    return fig


def plot_modality_pie(modality_results):
    """Pie chart of cell modality counts (matches MATLAB output)."""
    counts = {1: 0, 2: 0, 3: 0, -1: 0, 0: 0}
    for r in modality_results.values():
        counts[r['modality']] = counts.get(r['modality'], 0) + 1

    # MATLAB includes Unimodal, Bimodal, Multimodal in the chart
    # (omits "too few" and "non-modal" by convention — Brad's original keeps these out)
    labels  = []
    sizes   = []
    colors  = []

    if counts[1] > 0:
        labels.append(f'Unimodal\n({counts[1]})');   sizes.append(counts[1]); colors.append('red')
    if counts[2] > 0:
        labels.append(f'Bimodal\n({counts[2]})');    sizes.append(counts[2]); colors.append('blue')
    if counts[3] > 0:
        labels.append(f'Multimodal\n({counts[3]})'); sizes.append(counts[3]); colors.append('white')
    # Optionally include non-modal — MATLAB excludes by default, but useful for visibility
    if counts[-1] > 0:
        labels.append(f'Non-modal\n({counts[-1]})'); sizes.append(counts[-1]); colors.append('lightgray')

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                            startangle=90, counterclock=False,
                            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.set_title(f"Modality distribution (total = {sum(sizes)} cells)",
                 fontweight='bold')
    fig.tight_layout()
    return fig


def plot_individual_cell_histograms(firing_rate_per_phase, modality_results, phase_centers,
                                    n_per_category=6, save_dir=None):

    
    # Separate cells by modality
    unimodal_ids = [cid for cid, r in modality_results.items() if r['modality'] == 1]
    bimodal_ids = [cid for cid, r in modality_results.items() if r['modality'] == 2]
    
    phase_720 = np.concatenate([phase_centers, phase_centers + 360])
    
    # Unimodal cells 
    n_uni = min(len(unimodal_ids), n_per_category)
    if n_uni > 0:
        cols = min(3, n_uni)
        rows = int(np.ceil(n_uni / cols))
        fig_uni, axes_uni = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes_uni = np.atleast_2d(axes_uni).ravel()
        
        for i, cell_id in enumerate(unimodal_ids[:n_uni]):
            ax = axes_uni[i]
            data = firing_rate_per_phase[cell_id]
            result = modality_results[cell_id]
            
            raw_720 = np.concatenate([data['raw_rate'], data['raw_rate']])
            smooth_720 = np.concatenate([data['smooth_rate'], data['smooth_rate']])
            
            ax.bar(phase_720, raw_720, width=phase_centers[1]-phase_centers[0],
                    alpha=0.4, color='blue', edgecolor='black', linewidth=0.5)
            ax.plot(phase_720, smooth_720, 'r-', linewidth=2)
            
            # Mark peak
            for peak_phase in result['peak_phases']:
                ax.axvline(peak_phase, color='green', linestyle='--', alpha=0.7)
                ax.axvline(peak_phase + 360, color='green', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Theta Phase (°)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'Unimodal Cell {cell_id}')
            ax.set_xlim(0, 720)
            ax.axvline(360, color='gray', linestyle=':', alpha=0.5)
        
        # Turn off unused axes
        for i in range(n_uni, len(axes_uni)):
            axes_uni[i].axis('off')
        
        fig_uni.suptitle('Unimodal Cells', fontweight='bold')
        fig_uni.tight_layout()
    else:
        fig_uni = None
    
    # Bimodal cells 
    n_bi = min(len(bimodal_ids), n_per_category)
    if n_bi > 0:
        cols = min(3, n_bi)
        rows = int(np.ceil(n_bi / cols))
        fig_bi, axes_bi = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes_bi = np.atleast_2d(axes_bi).ravel()
        
        for i, cell_id in enumerate(bimodal_ids[:n_bi]):
            ax = axes_bi[i]
            data = firing_rate_per_phase[cell_id]
            result = modality_results[cell_id]
            
            raw_720 = np.concatenate([data['raw_rate'], data['raw_rate']])
            smooth_720 = np.concatenate([data['smooth_rate'], data['smooth_rate']])
            
            ax.bar(phase_720, raw_720, width=phase_centers[1]-phase_centers[0],
                    alpha=0.4, color='red', edgecolor='black', linewidth=0.5)
            ax.plot(phase_720, smooth_720, 'b-', linewidth=2)
            
            # Mark peaks
            for peak_phase in result['peak_phases']:
                ax.axvline(peak_phase, color='green', linestyle='--', alpha=0.7)
                ax.axvline(peak_phase + 360, color='green', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Theta Phase (°)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'Bimodal Cell {cell_id}')
            ax.set_xlim(0, 720)
            ax.axvline(360, color='gray', linestyle=':', alpha=0.5)
        
        # Turn off unused axes
        for i in range(n_bi, len(axes_bi)):
            axes_bi[i].axis('off')
        
        fig_bi.suptitle('Bimodal Cells', fontweight='bold')
        fig_bi.tight_layout()
    else:
        fig_bi = None
    
    # Save if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        if fig_uni is not None:
            fig_uni.savefig(save_dir / 'unimodal_cell_histograms.png', dpi=150, bbox_inches='tight')
        if fig_bi is not None:
            fig_bi.savefig(save_dir / 'bimodal_cell_histograms.png', dpi=150, bbox_inches='tight')
    
    return fig_uni, fig_bi
