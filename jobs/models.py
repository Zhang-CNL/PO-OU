import click
import os
import sys 
import json
import traceback
sys.path.append(os.path.realpath("..")) # Add hippocampalseq to path

from scipy.io import savemat

import hippocampalseq as hse
import hippocampalseq.preprocessing as hsep
import hippocampalseq.models as hsem

MODELS=['map', 'momentum', 'gridsearch']

def save_all_raw_data(
        raw_data: hsep.RawData,
        place_field_data: hsep.PlaceFields,
        theta_data: hsep.Theta,
        ripple_data: hsep.Ripples,
        environment_size: tuple,
        data_type: str
    ):
    # Just save EVERYTHING 
    savemat(os.path.join(rp, 'raw_data.mat'), {
        'raw_position':       raw_data.raw_position,
        'running_position':   raw_data.running_position,
        'excitatory_neurons': raw_data.excitatory_neurons,
        'inhibitory_neurons': raw_data.inhibitory_neurons,
        'environment_size':   environment_size
    }, do_compression=True)
    hseu.save_tsg_mat(os.path.join(rp, 'raw_spikes.mat'),
        raw_data.raw_spikes,
        do_compression=True
    )
    hseu.save_tsg_mat(os.path.join(rp, 'running_spikes.mat'),
        raw_data.running_spikes,
        do_compression=True
    )
    hseu.save_tsg_mat(os.path.join(rp, 'running_spike_info.mat'),
        raw_data.running_spike_info,
        do_compression=True
    )
    savemat(os.path.join(rp, 'place_field_data.mat'), {
        'place_fields': place_field_data.place_fields,
        'place_cell_ids': place_field_data.place_cell_ids
    }, do_compression=True)
    if data_type == 'theta':
        savemat(os.path.join(rp, 'theta_data.mat'), {
            'true_trajectory': theta_data.true_trajectory,
            'theta_spikes': theta_data.theta_spikes
        }, do_compression=True)
    elif data_type == 'replay':
        savemat(os.path.join(rp, 'ripple_data.mat'), {
            'ripple_spikes': ripple_data.ripple_spikes
        }, do_compression=True)

def save_results(values: hsem.StateSpaceResults, results_path: str):
    if isinstance(values, hsem.BayesianMAPResults):
        decoded_trajectories     = values.decoded_trajectories
        cumulative_probabilities = values.cumulative_probabilities
        trajectory_covariance    = []
        loglike                  = []
    elif isinstance(values, hsem.MomentumResults):
        decoded_trajectories     = [sm[:,:2] for sm in values.smoothed_mean]
        cumulative_probabilities = values.cumulative_probabilities
        trajectory_covariance    = [sc[:,:2,:2] for sc in values.smoothed_cov]
        loglike                  = values.loglike

    savemat(os.path.join(results_path, 'results.mat'), {
        'decoded_trajectories': decoded_trajectories,
        'cumulative_probabilities': cumulative_probabilities,
        'trajectory_covariance': trajectory_covariance,
        'loglike': loglike
    }, do_compression=True)

def run_model(
        model_selection: str,
        place_field_data: hsep.PlaceFields,
        dt: float,
        bin_size: int,
        spikemats: np.ndarray,
        environment_size: tuple,
        checkpoint_path: str,
    ):
    if model_selection == "map":
        model = hsem.BayesianMAP(
            place_field_data.place_fields,
            dt,
            bin_size
        )
    elif model_selection == "momentum":
        model = hsem.Momentum(
            place_field_data.place_fields,
            spikemats,
            dt,
            environment_size,
            bin_size,
        )
    elif model_selection == "gridsearch":
        pass
    print(f"Fitting {model_selection} model...")
    values = model.fit(
        spikemats,
        n_iter=10000,
        checkpoint_path=checkpoint_path,
    )
    return model,values


@click.command()
@click.option("--data-path", default="../data/")
@click.option("--results-path", default="../results")
@click.option("--data-type", default='theta', type=click.Choice(['theta', 'replay']))
@click.option("--delta-t-ms")
@click.option("--time-step-ms")
@click.option("--place-field-posterior", is_flag=True)
@click.option("--velocity-cutoff", default=10)
@click.option("--bin-size-cm", default=2)
@click.option("--skip-linear", is_flag=True)
@click.option("--rats", multiple=True, type=click.Choice(hsep.RAT_NAMES), default=hsep.RAT_NAMES)
@click.option("--checkpoint-path", default="../checkpoints/")
def main(
        data_path,
        results_path,
        data_type,
        delta_t_ms,
        time_step_ms,
        place_field_posterior,
        velocity_cutoff,
        bin_size_cm,
        skip_linear,
        rats,
        checkpoint_path,
    ):
    print(f"Processing rats: {rats}")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(results_path, 'config.json'), 'w') as f:
        js = json.dumps(
            {
                'data_type': data_type, 
                'delta_t_ms': delta_t_ms, 
                'time_step_ms': time_step_ms, 
                'place_field_posterior': place_field_posterior, 
                'velocity_cutoff': velocity_cutoff, 
                'bin_size_cm': bin_size_cm, 
            }, 
            indent=4, 
            sort_keys=True
        )
        f.write(js)
        print(js)

    for rat in rats:
        for session in os.listdir(os.path.join(data_path, rat)):
            rp = os.path.join(results_path, rat, session)
            if skip_linear and session.startswith('Linear'):
                continue

            os.makedirs(rp, exist_ok=True)
            print(f"Processing rat: {rat} session: {session}")
            if session.startswith('Linear'):
                env_size = None
            else:
                env_size = (0,0,200,200)

            (
                raw_data,
                place_field_data,
                theta_data,
                ripple_data
            ) = hsep.process_data(
                rat_name                      = rat,
                session                       = int(session[-1]),
                data_path                     = data_path,
                track_type                    = session[:-1],
                environment_size              = env_size,
                place_field_posterior         = place_field_posterior,
                bin_size_cm                   = bin_size_cm,
                velocity_cutoff               = velocity_cutoff,
                theta_time_window_ms          = delta_t_ms,
                theta_time_window_advance_ms  = time_step_ms,
                ripple_time_window_ms         = delta_t_ms,
                ripple_time_window_advance_ms = time_step_ms
            )
            print(f"Preprocessed {rat}--{session}")

            if session.startswith('Linear'):
                env_size = (
                    np.min(raw_data.raw_position['x']),
                    np.min(raw_data.raw_position['y']),
                    np.max(raw_data.raw_position['x']),
                    np.max(raw_data.raw_position['y'])
                )
            save_all_raw_data(
                raw_data,
                place_field_data,
                theta_data,
                ripple_data,
                env_size,
                data_type
            )
            if data_type == 'theta':
                spike_data = theta_data.theta_spikes
            elif data_type == 'replay':
                spike_data = ripple_data.ripple_spikes


            for model in MODELS:
                try:
                    model,values = fit_model(
                        model,
                        place_field_data,
                        delta_t_ms / 1000,
                        bin_size_cm,
                        spike_data,
                        env_size,
                        checkpoint_path
                    )
                except Exception as e:
                    print(traceback.format_exc(), file=sys.stderr)
                    print("\n"*5, file=sys.stderr)
                    print(f"Failed to fit {model} for {rat}--{session}--{data_type}. Error: {e}")
                    continue

                print(f"Finished fitting {model} for {rat}--{session}")

                save_results(
                    values,
                    os.path.join(rp, model)
                )



if __name__ == '__main__':
    main()