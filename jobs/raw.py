import os
import sys 
import time
import json
import click

from notebooks.examine_data import track_type
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))

import numpy as np
from dataclasses import asdict
from typing import Any
from pathlib import Path

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.utils as hseu
import hippocampalseq.plotting as hsepl
import hippocampalseq.analysis as hsea
import hippocampalseq.preprocessing as hsepp

def raw_plots(
        results_path: Path|str,
        track_type: str,
        raw_data: hse.RawData,
        place_field_data: hse.PlaceFields
    ):
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

def analyze_theta(
        raw_data: hse.RawData,
        place_field_data: hse.PlaceFields,
        theta_data: hse.Theta,
        parameters: dict[str, Any],
        results_path: Path|str,
    ):
    total_duration = raw_data.raw_position.time_support.tot_length('s')
    velocity_cutoff = parameters.get('velocity_cutoff', 10.0)

    (
        firing_rate_per_phase,
        phase_centers
    ) = hsea.calculate_phase_locking(
        theta_data.spikes_with_phase,
        total_duration,
        velocity_cutoff
    )
    modality_results = hsea.calculate_theta_modality(
        firing_rate_per_phase,
        theta_data.spikes_with_phase
    )
    population_statistics = hsea.calculate_population_firing_rates(
        firing_rate_per_phase,
        modality_results,
        raw_data.excitatory_neurons
    )
    eset = set(raw_data.excitatory_neurons)
    true_excit_in_phase = sorted(
        set(theta_data.spikes_with_phase.keys()) & eset
    )

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
        velocity_cutoff
    )

    hsepl.plot_phase_locked(
        firing_rate_per_phase,
        phase_centers,
        file_path=results_path,
        file_name="phase_locked.pdf"
    )
    plt.close()
    hsepl.plot_modality_classification(
        firing_rate_per_phase,
        modality_results,
        population_stats,
        phase_centers,
        file_path=results_path,
        file_name="modality_classification.pdf"
    )
    plt.close()
    hsepl.plot_modality_all_cells(
        firing_rate_per_phase,
        modality_results,
        phase_centers,
        file_path=results_path,
        file_name="modality_all_cells.pdf"
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
    return {
        'firing_rate_per_phase' : firing_rate_per_phase,
        'phase_centers'         : phase_centers,
        'modality_results'      : modality_results,
        'population_statistics' : population_stats,
        'pooled_cells'          : pooled_cells,
        'unimodal_cells'        : unimodal_cells,
        'bimodal_cells'         : bimodal_cells
    }


@click.command()
@click.option("--data-path", default="../data/")
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rats", multiple=True, type=click.Choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        data_path: str, 
        results_path: str, 
        run_config: str,
        rats: list[str]
    ):
    if not isinstance(rats, list):
        rats = [rats]
    data_path = Path(data_path)
    results_path = Path(results_path)

    os.makedirs(results_path, exist_ok=True)
    with open(Path(run_config), 'r') as f:
        parameters = json.loads(f.read())

    profile_name = parameters.get("name")
    results_path /= profile_name
    os.makedirs(results_path, exist_ok=True)

    with open(results_path / "config.json", 'w') as f:
        json.dump(parameters, f, indent=4)

    for rat in rats:
        rat_path = data_path / rat
        for session in os.listdir(rat_path):
            track_type = session[:-1]
            session_n  = int(session[-1])
            env_size = None if track_type == 'Linear' else hsep.EnvironmentSize(
                (0,200), (0,200)
            )
            session_path = rat_path / session

            if track_type not in parameters.get("session_types", ["Linear", "Open"]):
                continue

            print(f"Processing {rat} {session}")
            start = time.time()
            results_dir = results_path / rat / session
            os.makedirs(results_dir, exist_ok=True)
            (
                raw_data,
                place_field_data
            ) = hse.load_raw_data(
                data_path,
                rat,
                session_n,
                track_type,
                bin_size_cm       = parameters.get("bin_size_cm", 2.0),
                environment_size  = env_size,
                loading_kwargs    = parameters.get("loading_args", {}),
                placefield_kwargs = parameters.get("placefield_args", {})
            )
            hseio.save_to_mat2(
                results_dir / "raw_data.mat",
                {
                    'raw_data'         : raw_data,
                    'place_field_data' : place_field_data
                }
            )

            raw_plots(
                results_dir,
                track_type,
                raw_data,
                place_field_data
            )
            print(f"Raw analysis finished. Took {time.time() - start}")

            env_size = raw_data.environment_size

            if not parameters.get("ignore_theta", False):
                print("Preprocessing theta.")
                start = time.time()
                theta_data = hse.process_theta(
                    raw_data,
                    place_field_data,
                    velocity_cutoff = parameters.get("velocity_cutoff", 10.0),
                    theta_kwargs    = parameters.get("theta_args", {})
                )
                hseio.save_to_mat2(
                    results_dir / "theta_data.mat",
                    theta_data
                )
                print(f"Finished preprocessing theta. Took {time.time() - start}")

                print("Analyzing theta.")
                start = time.time()

                theta_analyzed = analyze_theta(
                    raw_data,
                    place_field_data,
                    theta_data,
                    parameters,
                    results_dir,
                )
                hseio.save_to_mat2(
                    results_dir / "theta_analyzed.mat",
                    theta_analyzed
                )
                print(f"Finished analyzing theta. Took {time.time() - start}")
                

            if not parameters.get("ignore_ripples", False):
                raise NotImplementedError

    print("Job completed")

if __name__ == "__main__":
    main()