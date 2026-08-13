import click
import os 
import sys 
import json 
sys.path.append(os.path.realpath("..")) # Add hippocampalseq to path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.plotting as hsepl
import hippocampalseq.utils as hseu
import hippocampalseq.analysis as hsea

def raw_analysis(results_path, track_type, raw_data):
    place_field_data = raw_data['place_field_data']
    raw_data = raw_data['raw_data']
    hsepl.plot_place_fields(
        place_field_data,
        raw_data,
        track_type,
        results_path
    ) 
    hsepl.plot_trajectory_with_velocity(
        raw_data.raw_position[['x','y']].values,
        raw_data.raw_position['Velocity'].values,
        raw_data.environment_size if track_type == 'Open' else None,
        file_path=results_path,
        file_name="true_trajectories.pdf"
    )

    hsepl.plot_lfp_data(
        raw_data.lfp_data,
        file_path=results_path,
        file_name="lfp_data.pdf"
    )
    
    if len(raw_data.environment_size) == 1:
        hsepl.plot_session_stitching(
            raw_data.running_position,
            raw_data.running_spikes,
            file_path=results_path,
            file_name="session_stitching.pdf"
        )

def theta_analysis(results_path, track_type, raw_data, theta_data, model_results, parameters):
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
    with PdfPages(os.path.join(results_path, "model_results.pdf")) as pdf:
        for i in range(len(theta_data.ground_truth)):
            fig = plt.figure(dpi=300)
            gt = theta_data.ground_truth[i]
            if track_type == 'Linear' and len(environment_size) == 1:
                    max_axis = 'x' if (np.max(gt['x']) - np.min(gt['x'])) > (np.max(gt['y']) - np.min(gt['y'])) else 'y'
                    true_trajectory = gt[max_axis].values
            else:
                true_trajectory = gt[['x', 'y']].values
            
            plt.subplot(3,2,1)
            hsepl.plot_trajectories(
                true_trajectory,
                environment_size=environment_size
            )
            plt.title("True trajectory")
            plt.subplot(3,2,2)
            times = gt.index.values
            sampling = parameters.get('theta_args')['time_window_ms'] / 1000
            times1 = np.arange(times[0], times[-1]+10*sampling, sampling)[:len(map_decoded['trajectory'][i])]
            hsepl.plot_trajectories(
                {
                    "True trajectory": true_trajectory,
                    "MAP": map_decoded['trajectory'][i],
                    "Momentum": momentum_decoded['trajectory'][i]
                },
                [
                    times,
                    times1,
                    times1
                ],
                environment_size=environment_size
            )
            plt.title("All trajectories")
            plt.subplot(3,2,3)
            hsepl.plot_trajectories(
                map_decoded['trajectory'][i],
                environment_size=environment_size
            )
            plt.title("MAP trajectory")
            plt.subplot(3,2,4)
            cumprob = map_decoded['cumprob'][i]
            if cumprob.ndim < 2:
                cumprob = cumprob[...,None]
            plt.imshow(
                cumprob.T,
                origin='lower',
                cmap='hot'
            )
            plt.title("MAP cumulative probability")
            plt.subplot(3,2,5)
            hsepl.plot_trajectories(
                momentum_decoded['trajectory'][i],
                environment_size=environment_size
            )
            plt.title(f"Momentum trajectory.\nAIC: {momentum_decoded['aic']:.2f}, BIC: {momentum_decoded['bic']:.2f}")
            plt.subplot(3,2,6)
            cumprob = momentum_decoded['cumprob'][i]
            if cumprob.ndim < 2:
                cumprob = cumprob[...,None]
            plt.imshow(
                cumprob.T,
                origin='lower',
                cmap='hot'
            )
            plt.title("Momentum cumulative probability")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def replay_analysis(results_path, track_type, raw_data, replay_data, parameters):
    pass

@click.command()
@click.option("--results-path", default="../results/")
@click.option("--run-config")
@click.option("--profile-name", default=None)
@click.option("--rats", multiple=True, type=click.Choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        results_path: str,
        run_config: str,
        profile_name: str,
        rats: list[str],
    ):
    with open(os.path.realpath(run_config), 'r') as f:
        parameters = json.loads(f.read())

    profile_name = parameters.get("name", profile_name)
    print(profile_name)
    results_path = os.path.join(results_path, profile_name)

    for rat in rats:
        rat_path = os.path.join(results_path, rat)
        for session in os.listdir(rat_path):
            results_dir = os.path.join(rat_path, session)
            track_type = session[:-1]
            session_n  = int(session[-1])
            print(f"Analyzing {rat} {track_type} {session_n}")

            raw_data = hseio.load_from_mat2(
                os.path.join(results_dir, 'raw_data.mat')
            )

            raw_analysis(
                results_dir,
                track_type,
                raw_data
            )

            theta_path = os.path.join(results_dir, 'theta_raw.mat')
            model_results_path = os.path.join(results_dir, 'model_theta_results.mat')
            if os.path.exists(model_results_path) and os.path.exists(theta_path):
                model_results = hseio.load_from_mat2(model_results_path)
                theta_data = hseio.load_from_mat2(theta_path)
                theta_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    theta_data,
                    model_results,
                    parameters
                )

            replay_path = os.path.join(results_dir, 'replay_data.mat')
            model_results_path = os.path.join(results_dir, 'model_swr_results.mat')
            if os.path.exists(model_results_path) and os.path.exists(replay_path):
                model_results = hseio.load_from_mat2(model_results_path)
                replay_data = hseio.load_from_mat2(replay_path)
                replay_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    replay_data,
                    parameters
                )
                
    print("Job completed")


if __name__ == '__main__':
    main()