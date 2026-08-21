import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as PdfPages

from typing import Any

import hippocampalseq as hse
import hippocampalseq.plotting as hsepl
import hippocampalseq.analysis as hsea
from . import utils

def theta_analysis(
        results_path: str, 
        track_type: str, 
        raw_data: dict[str, [hse.RawData, hse.PlaceFields]], 
        theta_data: dict[str, hse.Theta], 
        model_results: dict[str, Any], 
        parameters: dict[str, Any]
    ):
    place_field_data = raw_data['place_field_data']
    raw_data = raw_data['raw_data']
    theta_data = theta_data['theta_data']

    total_duration = raw_data.raw_position.time_support.tot_length('s')
    velocity_cutoff = parameters.get('velocity_cutoff', 10.0)

    firing_rate_per_phase,phase_centers = hsea.calculate_phase_locking(
        theta_data.spikes_with_phase,
        total_duration,
        velocity_cutoff
    )

    modality_results = hsea.classify_theta_modality(
        firing_rate_per_phase,
        theta_data.spikes_with_phase
    )

    population_stats = hsea.calculate_population_firing_rates(
        firing_rate_per_phase,
        modality_results,
        raw_data.excitatory_neurons,
    )

    eset = set(raw_data.excitatory_neurons)
    true_excit_in_phase = sorted(set(theta_data.spikes_with_phase.keys()) & eset)

    (
        pooled_cells,
        unimodal_cells,
        bimodal_cells,
    ) = hsea.classify_place_cell_modality(
        place_field_data.place_fields,
        place_field_data.place_cell_ids,
        place_field_data.position_hist,
        theta_data.spikes_with_phase,
        modality_results,
        true_excit_in_phase,
        velocity_cutoff,
    )

    hsepl.plot_phase_locked(
        firing_rate_per_phase,
        phase_centers,
        file_path=results_path,
        file_name="phase_locked.pdf"
    )

    hsepl.plot_modality_classification(
        firing_rate_per_phase,
        modality_results,
        population_stats,
        phase_centers,
        file_path=results_path,
        file_name="modality_classification.pdf"
    )

    hsepl.plot_modality_all_cells(
        firing_rate_per_phase,
        modality_results,
        phase_centers,
        file_path=results_path,
        file_name="modality_all_cells.pdf"
    )
    plt.close()

    hsepl.plot_place_field_comparison(
        unimodal_cells,
        bimodal_cells,
        file_path=results_path,
        file_name="place_field_comparison.pdf"
    )
    plt.close()

    hsepl.plot_unimodal_bimodal_summary(
        modality_results,
        population_stats,
        phase_centers,
        unimodal_cells,
        bimodal_cells,
        raw_data.excitatory_neurons,
        file_path=results_path,
        file_name="unimodal_bimodal_summary.pdf"
    )
    plt.close()

    hsepl.plot_modality_overlay(
        population_stats, 
        phase_centers,
        file_path=results_path,
        file_name="modality_overlay.pdf"
    )
    plt.close()

    hsepl.plot_modality_pie(
        modality_results,
        file_path=results_path,
        file_name="modality_pie.pdf"
    )
    plt.close()

    hsepl.plot_theta_phase_assignment(
        theta_data.spikes_with_phase,
        theta_data.lfp_data,
        theta_data.trough_indices,
        raw_data.excitatory_neurons,
        time_window=2,
        file_path=results_path,
        file_name="theta_phase_assignment.pdf"
    )
    plt.close()

    hsepl.plot_cell_phase_polar(
        theta_data.spikes_with_phase,
        file_path=results_path,
        file_name="cell_phase_polar.pdf"
    )
    plt.close()

    hsepl.plot_theta_lfp_segment(
        theta_data.lfp_data,
        theta_data.trough_times,
        theta_data.trough_indices,
        2.0,
        file_path=results_path,
        file_name="theta_lfp_segment.pdf"
    )
    plt.close()

    hsepl.plot_theta_cycle_dist(
        theta_data.lfp_data,
        file_path=results_path,
        file_name="theta_cycle_dist.pdf"
    )
    plt.close()



    environment_size = raw_data.environment_size
    map_decoded = model_results['map']
    momentum_decoded = model_results['momentum']
    momentum_v_true_decoded = model_results['momentum_v_true']
    momentum_v_pred_decoded = model_results['momentum_v_pred']
    ndim = len(environment_size)
    nmodels = len(model_results) + 1
    with PdfPages(os.path.join(results_path, "model_results.pdf")) as pdf:
        for i in range(len(theta_data.ground_truth)):
            fig = plt.figure(figsize=(50,50), fdpi=300)
            gt = theta_data.ground_truth[i]
            true_trajectory = utils.extract_trajectory(
                gt,
                environment_size,
                track_type
            )

            momentum_trajectory = momentum_decoded['decoded_trajectories'][i][:,ndim:]
            momentum_true_trajectory = momentum_v_true_decoded['decoded_velocities'][i][:,ndim:]
            momentum_pred_trajectory = momentum_v_pred_decoded['decoded_velocities'][i][:,ndim:]
            
            plt.subplot(nmodels,2,1)
            hsepl.plot_trajectories(
                true_trajectory,
                environment_size=environment_size
            )
            plt.title("True trajectory")
            plt.subplot(nmodels,2,2)
            times = gt.index.values
            sampling = parameters.get('theta_args')['time_window_s']
            times1 = np.arange(times[0], times[-1]+10*sampling, sampling)[:len(map_decoded['decoded_trajectories'][i])]
            hsepl.plot_trajectories(
                {
                    "True trajectory": true_trajectory,
                    "MAP": map_decoded['decoded_trajectories'][i],
                    "Momentum": momentum_trajectory,
                    "Momentum True Velocity": momentum_true_trajectory,
                    "Momentum Predicted Velocity": momentum_pred_trajectory
                },
                [times] + 4*[times1],
                environment_size=environment_size
            )
            plt.title("All trajectories")

            utils.model_result_plot(
                nmodels,2,3,
                map_decoded['decoded_trajectories'][i],
                environment_size,
                map_decoded['cumulative_probabilities'][i],
                "MAP",
            )

            utils.model_result_plot(
                nmodels,2,5,
                momentum_trajectory,
                environment_size,
                momentum_decoded['cumulative_probabilities'][i],
                "Momentum",
                momentum_decoded['aic'],
                momentum_decoded['bic']
            )

            utils.model_result_plot(
                nmodels,2,7,
                momentum_true_trajectory,
                environment_size,
                momentum_v_true_decoded['cumulative_probabilities'][i],
                "Momentum with observed true velocity",
                momentum_v_true_decoded['aic'],
                momentum_v_true_decoded['bic']
            )
            
            utils.model_result_plot(
                nmodels,2,9,
                momentum_pred_trajectory,
                environment_size,
                momentum_v_pred_decoded['cumulative_probabilities'][i],
                "Momentum with predicted true velocity",
                momentum_v_pred_decoded['aic'],
                momentum_v_pred_decoded['bic']
            )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
