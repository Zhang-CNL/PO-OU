import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
from utils.Theta_Phase.theta_sequence_decoding import decode_theta_sequences, decode_unimodal_vs_bimodal


def get_cells_by_modality(modality_results, target_modality):
    """Return sorted list of cell IDs by modality."""
    if isinstance(target_modality, int):
        target_modality = [target_modality]
    return sorted([
        cid for cid, info in modality_results.items()
        if info['modality'] in target_modality
    ])
    
    
def filter_predecode_by_cells(predecode, included_cells):
    """
    Zero out spikes from cells NOT in `included_cells`, then drop windows that
    have no remaining spikes from the included subset.
    """
    decoding_spike_index = predecode['decoding_spike_index'].copy()

    # Set entries that aren't in the included cell set to 0
    # (zeros stay zero — this is idempotent for them)
    included_mask = np.isin(decoding_spike_index, included_cells)
    decoding_spike_index[~included_mask] = 0

    # Keep only original valid windows that still have at least one spike
    has_spikes_per_window = np.any(decoding_spike_index > 0, axis=0)
    original_window_index = predecode['decoding_window_index']
    new_window_index = original_window_index[
        has_spikes_per_window[original_window_index]
    ]

    return {
        **predecode,
        'decoding_spike_index': decoding_spike_index,
        'decoding_window_index': new_window_index,
    }

def decode_by_modality(predecode, field_results, modality_results,
                    initial_variables):
    """
    Decode theta windows four ways:
    1. ALL excitatory cells
    2. UNIMODAL only         (product = uni cells, sum = ALL cells)
    3. BIMODAL only          (product = bi cells,  sum = ALL cells)
    4. UNIMODAL vs BIMODAL   (product = uni or bi, sum = same subset, JOINT norm)
    
    1-3 mirror MATLAB IRFS_DECODE_THETA_SEQUENCES_WITH_*_CELLS.
    4 mirrors MATLAB IRFS_DECODE_THETA_SEQUENCES_WITH_UNIMODAL_VS_BIMODAL_CELLS,
    which is analogous to IN/OUT field decoding on linear tracks and quantifies
    the relative contribution of each population to representing position.
    
    I store the Uni vs bimodal (#4) but it's not really necessary to our analysis specifically
    This is because in the original paper, they  wanted to see how much of the sweep is being encoded by each cell type
    """
    results    = {}
    predecodes = {}

    # 1. ALL cells
    print("\n=== Decoding with ALL cells ===")
    results['all'] = decode_theta_sequences(predecode, field_results,
                                            initial_variables)
    results['all']['n_cells']  = len(field_results['cell_ids'])
    results['all']['cell_ids'] = list(field_results['cell_ids'])
    predecodes['all'] = predecode

    # 2. UNIMODAL only (sum over all cells, like MATLAB)
    print("\n=== Decoding with UNIMODAL cells ===")
    uni_cells = get_cells_by_modality(modality_results, target_modality=1)
    print(f"  Found {len(uni_cells)} unimodal cells")
    if len(uni_cells) > 0:
        uni_predecode = filter_predecode_by_cells(predecode, uni_cells)
        print(f"  Windows after filter: "
            f"{len(uni_predecode['decoding_window_index'])} / "
            f"{len(predecode['decoding_window_index'])}")
        results['unimodal'] = decode_theta_sequences(uni_predecode,
                                                    field_results,
                                                    initial_variables)
        results['unimodal']['n_cells']  = len(uni_cells)
        results['unimodal']['cell_ids'] = uni_cells
        predecodes['unimodal'] = uni_predecode
    else:
        results['unimodal']    = None
        predecodes['unimodal'] = None

    # 3. BIMODAL only (sum over all cells, like MATLAB)
    print("\n=== Decoding with BIMODAL cells ===")
    bi_cells = get_cells_by_modality(modality_results, target_modality=2)
    print(f"  Found {len(bi_cells)} bimodal cells")
    if len(bi_cells) > 0:
        bi_predecode = filter_predecode_by_cells(predecode, bi_cells)
        print(f"  Windows after filter: "
            f"{len(bi_predecode['decoding_window_index'])} / "
            f"{len(predecode['decoding_window_index'])}")
        results['bimodal'] = decode_theta_sequences(bi_predecode,
                                                    field_results,
                                                    initial_variables)
        results['bimodal']['n_cells']  = len(bi_cells)
        results['bimodal']['cell_ids'] = bi_cells
        predecodes['bimodal'] = bi_predecode
    else:
        results['bimodal']    = None
        predecodes['bimodal'] = None

    # 4. UNIMODAL vs BIMODAL (jointly normalized, mimicking IN/OUT decoding which I don't actually do since it's not the goal of our paper).  IF you want to look at goal-directed navigation, you will want to break up the UP/DOWN linear runs
    print("\n=== Decoding UNIMODAL vs BIMODAL (joint normalization) ===")
    print(f"  {len(uni_cells)} unimodal, {len(bi_cells)} bimodal cells")
    if len(uni_cells) > 0 and len(bi_cells) > 0:
        results['uni_vs_bi'] = decode_unimodal_vs_bimodal(
            predecode, field_results, uni_cells, bi_cells, initial_variables
        )
        predecodes['uni_vs_bi'] = predecode   # uses full predecode (all valid windows)
    else:
        print("  Need both unimodal and bimodal cells; skipping.")
        results['uni_vs_bi']    = None
        predecodes['uni_vs_bi'] = None

    return results, predecodes
        
