import os
import sys 
import time
import json
import click
import traceback
sys.path.append(os.path.realpath(".."))

import numpy as np
import scipy.io as sio
from dataclasses import asdict
from typing import Any

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.preprocessing as hsepp
import hippocampalseq.models as hsem

def save_raw_data(
        save_name: str,
        raw_data: hse.RawData,
        place_field_data: hse.PlaceFields,
        theta_data: hse.Theta,
        ripple_data: hse.Replay,
    ):
    sio.savemat(
        save_name,
        {
            'raw_data'         : asdict(raw_data),
            'place_field_data' : asdict(place_field_data),
            'theta_data'       : asdict(theta_data),
            'ripple_data'      : asdict(ripple_data)
        },
        do_compression = True
    )

def run_models_over_data(
        save_name: str,
        parameters: dict[str, Any],
        place_fields: np.ndarray,
        spikes: np.ndarray,
        dt: float,
        bin_size: int,
        environment_size: list[tuple[int,...]],
    ):
    start = time.time()
    map_model = hsem.BayesianMAP(
        place_fields,
        dt,
        bin_size
    )
    map_decoded = map_model.fit(spikes)
    end = time.time()
    print(f"Completed MAP decoding. {end-start} seconds")

    start = time.time()
    momentum_model = hsem.Momentum(
        place_fields,
        spikes,
        dt,
        environment_size,
        parameters.get('bin_size', 2.0),
        parameters.get('seed', 42)
    )
    momentum_decoded = momentum_model.fit(
        None,
        n_iter    = parameters.get('n_iter', 10000),
        optimizer = parameters.get('optimizer', 'Adam'),
        lr        = parameters.get('lr', .01),
    )
    end = time.time()
    print(f"Completed momentum decoding. {end-start} seconds")

    smoothed_mean = [
        sm[:,:2].numpy() for sm in momentum_decoded.smoothed_mean
    ]
    smoothed_cov = [
        sc[:,:2,:2].numpy() for sc in momentum_decoded.smoothed_cov
    ]
    sio.savemat(save_name,
        {
            'dt'               : dt,
            'environment_size' : environment_size,
            'bin_size'         : bin_size,
            'map': {
                'trajectory' : map_decoded.decoded_trajectories,
                'cumprob'    : map_decoded.cumulative_probabilities,
            },
            'momentum': {
                'trajectory' : smoothed_mean,
                'cov'        : smoothed_cov,
                'cumprob'    : momentum_decoded.cumulative_probabilities.numpy(),
                'loglike'    : momentum_decoded.loglike_full,
                'aic'        : momentum_decoded.aic,
                'bic'        : momentum_decoded.bic
            }
        },
        do_compression=True
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

    for rat in rats:
        rat_path = os.path.join(data_path, rat)
        for session in os.listdir(rat_path):
            print(f"Started working on rat {rat}, session {session}")
            if rat == 'Naga' and session == 'Open1':
                print("Missing LFP data for Naga Open1. Skipping.")
                continue
            if rat == 'Ettin':
                print("Skipping Ettin. Too noisy.")
                continue

            results_dir = os.path.join(rat_path, session)
            os.makedirs(results_dir, exist_ok=True)

            track_type = session[:-1]
            session_n  = int(session[-1])
            env_size   = None if track_type == 'Linear' else [(0,200),(0,200)]

            if track_type not in parameters.get("session_types", ["Linear", "Open"]):
                continue

            (
                raw_data,
                place_field_data,
                theta_data,
                ripple_data
            ) = hse.load_and_preprocess(
                data_path,
                rat,
                session_n,
                track_type        = track_type,
                environment_size  = env_size,
                bin_size_cm       = parameters.get('bin_size_cm', 2),

                placefield_kwargs = parameters.get('placefield_args', {}),
                loading_kwargs    = parameters.get('loading_args', {}),
                theta_kwargs      = parameters.get('theta_args', {}),
                ripple_kwargs     = parameters.get('ripple_args', {})
            )

            save_raw_data(
                os.path.join(results_dir, 'raw_data.mat'),
                raw_data,
                place_field_data,
                theta_data,
                ripple_data
            )

            print(f"Finished preprocessing {rat} {session}")
            
            if track_type == 'Linear':
                env_size = raw_data.environment_size

            place_fields = place_field_data.place_fields[place_field_data.place_cell_ids]

            try:
                if not parameters.get("ignore_theta", False):
                    run_models_over_data(
                        os.path.join(results_dir, 'model_theta_results.mat'),
                        parameters,
                        place_fields,
                        theta_data.spikes,
                        parameters.get('theta_time_window_ms', 60.0)/1000,
                        int(parameters.get('bin_size_cm', 2)),
                        env_size
                    )
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("-"*100, file=sys.stderr)
                print(f"\nFailed to decode theta for {rat} {session}. {e}", file=sys.stderr)
            try:
                if not parameters.get("ignore_ripples", False):
                    run_models_over_data(
                        os.path.join(results_dir, 'model_swr_results.mat'),
                        parameters,
                        place_fields,
                        ripple_data.spikes,
                        parameters.get('ripple_time_window_ms', 5.0)/1000,
                        int(parameters.get('bin_size_cm', 2)),
                        env_size
                    )
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("-"*100, file=sys.stderr)
                print(f"\nFailed to decode ripple for {rat} {session}. {e}", file=sys.stderr)


if __name__ == "__main__":
    main()