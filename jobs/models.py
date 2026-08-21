import os
import uuid
import sys 
import time
import json
import click
import traceback
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))

import numpy as np
from dataclasses import asdict
from typing import Any

import hippocampalseq as hse
import hippocampalseq.utils as hseu
import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsepp
import hippocampalseq.models as hsem

import analyze

def run_model(
        model_name: str,
        model_kwargs: dict[str, Any],
        model_fit_kwargs: dict[str, Any] = {},
    ):

    models = {
        'MAP'             : hsem.BayesianMAP,
        'Momentum'        : hsem.Momentum,
        'MomentumVelocity': hsem.MomentumVelocity,
    }
    model = models[model_name](**model_kwargs)
    return model.fit(**model_fit_kwargs)

def run_models_over_data(
        save_name: str,
        parameters: dict[str, Any],
        place_fields: np.ndarray,
        raw_data: hse.RawData,
        dataset: hse.Theta|hse.Replay,
        dt: float,
    ):
    bin_size = parameters.get('bin_size', 2.0)
    start = time.time()
    map_decoded = run_model('MAP',
        {
            'place_fields': place_fields, 
            'dt': dt, 
            'bin_size': bin_size
        },
        {'X': spikes}
    )
    end = time.time()
    print(f"Completed MAP decoding. {end-start} seconds")

    save_dict = {
        'map': asdict(map_decoded)
    }

    momentum_kwargs = {
        'place_fields'     : place_fields, 
        'spikes'           : spikes,
        'dt'               : dt, 
        'environment_size' : raw_data.environment_size,
        'bin_size'         : bin_size,
        'seed'             : parameters.get('seed', 42)
    }

    start = time.time()
    momentum_decoded = run_model('Momentum',
        momentum_kwargs,
        {
        }
    )
    end = time.time()
    print(f"Completed momentum decoding. {end-start} seconds")

    save_dict['momentum'] = {
        'decoded_trajectories' : momentum_decoded.smoothed_mean,
        'cov'             : momentum_decoded.smoothed_cov,
        'cumulative_probabilities': momentum_decoded.cumulative_probabilities,
        'loglike' : momentum_decoded.loglike_full,
        'aic'     : momentum_decoded.aic,
        'bic'     : momentum_decoded.bic
    }

    if isinstance(dataset, hse.Theta):
        start = time.time()
        if len(raw_data.environment_size) == 1:
            max_axis = analyze.get_major_axis(raw_data.raw_position)
            true_velocity = [v[f'V_{max_axis}'].values for v in theta_data.ground_truth]
        else:
            true_velocity = [v[['V_x', 'V_y']].values for v in theta_data.ground_truth]

        momentum_v_true_decoded = run_model('MomentumVelocity',
            momentum_kwargs | {'velocity_type': 'true'},
            {
                'X': true_velocity
            }
        )
        end = time.time()
        print(f"Completed momentum velocity decoding. {end-start} seconds")

        save_dict['momentum_v_true'] = {
            'decoded_trajectories' : momentum_v_true_decoded.smoothed_mean,
            'cov'             : momentum_v_true_decoded.smoothed_cov,
            'cumulative_probabilities': momentum_v_true_decoded.cumulative_probabilities,
            'loglike' : momentum_v_true_decoded.loglike_full,
            'aic'     : momentum_v_true_decoded.aic,
            'bic'     : momentum_v_true_decoded.bic
        }

        start = time.time()
        momentum_v_pred_decoded = run_model('MomentumVelocity',
            momentum_kwargs | {'velocity_type': 'observed'},
        )
        end = time.time()
        print(f"Completed momentum velocity decoding. {end-start} seconds")

        save_dict['momentum_v_pred'] = {
            'decoded_trajectories' : momentum_v_pred_decoded.smoothed_mean,
            'cov'             : momentum_v_pred_decoded.smoothed_cov,
            'cumulative_probabilities': momentum_v_pred_decoded.cumulative_probabilities,
            'loglike' : momentum_v_pred_decoded.loglike_full,
            'aic'     : momentum_v_pred_decoded.aic,
            'bic'     : momentum_v_pred_decoded.bic
        }

    hseio.save_to_mat2(save_name,
        save_dict
    )

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
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.realpath(run_config), 'r') as f:
        parameters = json.loads(f.read())

    profile_name = parameters.get("name", str(uuid.uuid4()))
    results_path = os.path.join(results_path, profile_name)
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, "config.json"), 'w') as f:
        json.dump(parameters, f, indent=4)

    for rat in rats:
        rat_path = os.path.join(data_path, rat)
        for session in os.listdir(rat_path):
            track_type = session[:-1]
            session_n  = int(session[-1])
            env_size   = None if track_type == 'Linear' else [(0,200),(0,200)]

            if rat == "Janni" and track_type == "Open":
                print("Skipping Janni Open tracks (broken data)")
                continue

            if track_type not in parameters.get("session_types", ["Linear", "Open"]):
                continue

            print(f"Started working on rat {rat}, session {session}")
            results_dir = os.path.join(results_path, rat, session)
            os.makedirs(results_dir, exist_ok=True)

            (
                raw_data,
                place_field_data
            ) = hse.load_raw_data(
                data_path,
                rat,
                session_n,
                track_type,
                bin_size_cm       = parameters.get('bin_size_cm', 2.0),
                environment_size  = env_size,
                loading_kwargs    = parameters.get('loading_args', {}),
                placefield_kwargs = parameters.get('placefield_args', {})
            )

            hseio.save_to_mat2(
                os.path.join(results_dir, 'raw_data.mat'),
                {
                    'raw_data'         : raw_data,
                    'place_field_data' : place_field_data
                }
            )

            print(f"Finished loading {rat} {session}")
            
            if track_type == 'Linear':
                env_size = raw_data.environment_size

            place_fields = place_field_data.place_fields[place_field_data.place_cell_ids]

            try:
                if not parameters.get("ignore_theta", False):
                    theta_data = hse.process_theta(
                        raw_data,
                        place_field_data,
                        velocity_cutoff = parameters.get('velocity_cutoff', 10.0),
                        theta_kwargs    = parameters.get('theta_args', {})
                    )
                    hseio.save_to_mat2(
                        os.path.join(results_dir, 'theta_raw.mat'),
                        {"theta_data": theta_data },
                    )
                    run_models_over_data(
                        os.path.join(results_dir, 'model_theta_results.mat'),
                        parameters,
                        place_fields,
                        theta_data.spikes,
                        parameters.get('theta_time_window_s', 60.0/1000),
                        int(parameters.get('bin_size_cm', 2)),
                        raw_data.environment_size
                    )
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("-"*100, file=sys.stderr)
                print(f"\nFailed to decode theta for {rat} {session}. {e}", file=sys.stderr)


            try:
                if not parameters.get("ignore_ripples", False):
                    raise NotImplementedError("Ripple preprocessing not implemented yet.")
                    run_models_over_data(
                        os.path.join(results_dir, 'model_swr_results.mat'),
                        parameters,
                        place_fields,
                        ripple_data.spikes,
                        parameters.get('ripple_time_window_s', 5.0/1000),
                        int(parameters.get('bin_size_cm', 2)),
                        raw_data.environment_size
                    )
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("-"*100, file=sys.stderr)
                print(f"\nFailed to decode ripple for {rat} {session}. {e}", file=sys.stderr)

    print("Job completed")

if __name__ == "__main__":
    main()