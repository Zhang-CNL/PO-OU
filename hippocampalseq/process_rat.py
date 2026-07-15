import time

import hippocampalseq.analysis as hsea
from .load_rat import RawData, PlaceFields, Theta, Replay 

def process_ratdata(
        raw_data: RawData,
        placefield_data: PlaceFields,
        theta_data: Theta,
        replay_data: Replay,
        velocity_cutoff: float = 5.0,
        phase_bin_size_deg: int = 10,
        gaussian_std: float = 12,
        limit_analysis_by_theta_length: bool = True,
        theta_length_s: tuple[float,float] = (0.08,0.16),
        minimum_spike_count: int = 100,
        rayleigh_p_cutoff: float = 0.05,
        min_field_fractions: float = 0.2,
        min_contiguous_bins: int = 20,
    ):
    total_duration = raw_data.raw_position.time_support.tot_length('s')

    begin = time.time()
    firing_rate_per_phase,phase_centers = hsea.calculate_phase_locking(
        theta_data.spikes_with_phase,
        total_duration,
        velocity_cutoff,
        phase_bin_size_deg,
        gaussian_std,
        limit_analysis_by_theta_length,
        theta_length_s,
        minimum_spike_count
    )
    print(f"Calculating phase locking took {time.time() - begin} seconds")

    begin = time.time()
    modality_results = hsea.classify_theta_modality(
        firing_rate_per_phase,
        theta_data.spikes_with_phase,
        phase_bin_size_deg,
        rayleigh_p_cutoff
    )
    print(f"Classifying theta modality took {time.time() - begin} seconds")

    begin = time.time()
    population_stats = hsea.calculate_population_firing_rates(
        firing_rate_per_phase,
        modality_results,
        raw_data.excitatory_neurons
    )
    print(f"Calculating population firing rates took {time.time() - begin} seconds")

    begin = time.time()
    unimodal_cells,bimodal_cells = hsea.classify_place_cell_modality(
        place_field_data.place_fields,
        place_field_data.place_cell_ids,
        place_field_data.position_hist,
        theta_data.spikes_with_phase,
        modality_results,
        raw_data.excitatory_neurons,
        velocity_cutoff,
        min_field_fractions,
        min_contiguous_bins
    )
    print(f"Classifying place cell modalities took {time.time() - begin} seconds")
    print(f"\nPlace Field Properties Summary:")
    print(f"  Unimodal cells analyzed: {len(unimodal_cells['mean_field_size'])}")
    print(f"  Bimodal cells analyzed:  {len(bimodal_cells['mean_field_size'])}")