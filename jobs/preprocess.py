import click 
import sys
import json
import os
sys.path.append(os.path.realpath("."))
sys.path.append(os.path.realpath(".."))

from pathlib import Path

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsep

@click.command()
@click.option("--data-path", default="../data/")
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rat", type=click.Choice(hseio.RAT_NAMES))
def main(data_path: str, results_path: str, run_config: str, rat: str):
    data_path    = Path(data_path)
    results_path = Path(results_path)
    run_config   = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())

    os.makedirs(results_path, exist_ok=True)
    with open(results_path / 'config.json', 'w') as f:
        json.dump(parameters, f, indent=4) # Save this for examination later

    for session in os.listdir(data_path / rat):
        track_type = session[:-1]
        session_num = int(session[-1])
        if track_type not in parameters.get("session_types", ["Linear", "Open"]):
            continue

        rat_data_path = data_path / rat / session
        os.makedirs(rat_data_path, exists_ok=True)

        environment_size = None
        if track_type == 'Open':
            environment_size = hsep.EnvironmentSize((0,200),(0,200))

        print(f"Processing {rat} {session}")

        processed_values = {}

        (
            raw_data,
            place_field_data
        ) = hse.load_raw_data(
            data_path,
            rat,
            session_num,
            track_type,
            environment_size=environment_size,
            bin_size_cm=parameters.get("bin_size", 2),
            loading_kwargs=parameters.get("loading_args", {}),
            placefield_kwargs=parameters.get("placefield_args", {})
        )

        processed_values["raw_data"] = raw_data
        processed_values["placefield_data"] = place_field_data

        if not parameters.get("ignore_theta", False):
            theta_data = hse.process_data(
                raw_data,
                place_field_data,
                theta_kwargs=parameters.get("theta_args", {})
            )
            processed_values["theta_data"] = theta_data

        if not parameters.get("ignore_ripples", False):
            raise NotImplementedError

        compress_pickle.dump(
            processed_values,
            results_path / rat / session / "preprocessed_data.pkl.gz"
        )

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

if __name__ == '__main__':
    main()