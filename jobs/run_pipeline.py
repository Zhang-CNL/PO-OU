import os
import sys 
import click
import glob
sys.path.append(os.path.realpath(".."))

from typing import List
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import multivariate_normal
import json
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

import hippocampalseq as hse
import hippocampalseq.preprocessing as hsep
import hippocampalseq.models as hsem 
import hippocampalseq.utils as hseu
import hippocampalseq.plotting as hsepl


def run_model(
        model_selection: str, 
        place_field_data: hsep.PlaceFields, 
        dt: float, 
        bin_size: int, 
        spikemats: np.ndarray, 
        environment_size: tuple,
        checkpoint_path: str,
        approximation_method: str,
        normalize: bool
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
            approximation_method=approximation_method
        )
    elif model_selection == "gridsearch":
        pass
    print("Constructed model. Fitting...")
    values = model.fit(
        spikemats,
        n_iter=10000,
        normalize=normalize,
        checkpoint_path=checkpoint_path
    )
    return model,values

def plot_place_fields(
        place_field_data: hsep.PlaceFields, 
        raw_data: nap.TsdFrame, 
        track_type: str, 
        results_path: str 
    ):
    with PdfPages(os.path.join(results_path, "place_fields.pdf")) as pdf:
        for i in range(len(place_field_data.place_fields)):
            fig,ax = plt.subplots(1,2,figsize=(20,10), dpi=300)
            plt.title(f"Place cell {i}")
            if track_type == 'Open':
                hsepl.plot_open_placefields(
                    place_field_data.place_fields,
                    pfs=[i],
                    show_titles=False,
                    ax=ax[0],
                )
                esize = (0,0,200,200)
            elif track_type == 'Linear':
                hsepl.plot_linear_placefields(
                    place_field_data.place_fields,
                    pfs=[i],
                    ax=ax[0]
                )
                esize=None
            hsepl.plot_spikemat_position_aligned(
                raw_data.running_spike_info,
                raw_data.raw_position,
                place_field_data.place_cell_ids, 
                environment_size=esize,
                cell_selection=[i],
                ax=ax[1]
            )
            pdf.savefig()
            plt.close()

        if track_type == 'Open':
            hsepl.plot_open_placefields(
                place_field_data.place_fields,
                show_titles=False,
            )
        elif track_type == 'Linear':
            hsepl.plot_linear_placefields(
                place_field_data.place_fields
            )
        pdf.savefig()
        plt.close(fig)

def plot_theta(
        values: hsem.StateSpaceResults, 
        true_trajectories: List[np.ndarray],
        track_type: str, 
        results_path: str
    ):
    if isinstance(values, hsem.BayesianMAPResults):
        trajectories = values.decoded_trajectories
        negloglike   = []
    elif isinstance(values, hsem.MomentumResults):
        trajectories = [sm[:,:2] for sm in values.smoothed_mean]
        negloglike   = values.negloglike
    with PdfPages(os.path.join(results_path, "theta.pdf")) as pdf:
        for i in range(len(true_trajectories)):
            fig,axs = plt.subplots(1,3+(1 if len(negloglike) > 0 else 0), figsize=(20,5), dpi=300)
            hsepl.plot_trajectories(
                [true_trajectories[i]],
                ax=axs[0]
            )
            hsepl.plot_trajectories(
                [trajectories[i]],
                ax=axs[1]
            )
            axs[2].imshow(
                values.cumulative_probabilities[i], 
                cmap='hot', 
                aspect='auto', 
                origin='lower'
            )
            if len(negloglike) > 0:
                axs[3].plot(negloglike)
                axs[3].set_title("Negative log likelihood")
            pdf.savefig()
            plt.close(fig)

def plot_replay(
        values: hsem.StateSpaceResults, 
        results_path: str
    ):
    if isinstance(values, hsem.MomentumResults):
        trajectory = [sm[:,:2] for sm in values.smoothed_mean]
        negloglike = values.negloglike
    elif isinstance(values, hsem.BayesianMAPResults):
        trajectory = values.decoded_trajectories
        negloglike = []
    with PdfPages(os.path.join(results_path, "replay.pdf")) as pdf:
        for i in range(len(trajectory)):
            fig,axs = plt.subplots(1,2+(1 if len(negloglike) > 0 else 0), figsize=(15,5), dpi=300)
            hsepl.plot_trajectories(
                [trajectory[i]],
                ax=axs[0]
            )
            axs[1].imshow(
                values.cumulative_probabilities[i], 
                cmap='hot', 
                aspect='auto', 
                origin='lower'
            )
            if len(negloglike) > 0:
                axs[2].set_title("Negative log likelihood")
                axs[2].plot(negloglike)
            pdf.savefig()
            plt.close(fig)

def plot_model_approximations(
        model: hsem.StateSpace,
        results_path: str
    ):
    if not isinstance(model, hsem.Momentum):
        return 
    X = np.arange(model.environment_size[0], model.environment_size[2])
    Y = np.arange(model.environment_size[1], model.environment_size[3])
    X,Y = np.meshgrid(X,Y)
    with PdfPages(os.path.join(results_path, "model_approximations.pdf")) as pdf:
        for i in range(len(model.emission_probabilities)):
            T = model.emission_probabilities[i].shape[0]
            fig,axs = plt.subplots(T,2, figsize=(10,5*(T+1)), dpi=300)
            # Plot kldiv
            for t in range(T):
                axs[t,0].imshow(
                    model.emission_probabilities[i][t], 
                    cmap='hot', 
                    aspect='auto', 
                    origin='lower'
                )
                Mean = model.approximate_mean[i][t].numpy()
                Cov = model.approximate_covariance[i][t].numpy()
                mvn = multivariate_normal(mean=Mean.ravel(), cov=Cov)
                Z = mvn.pdf(np.column_stack([X.ravel(), Y.ravel()]))
                Z = Z.reshape(X.shape) / np.sum(Z)
                axs[t,1].contourf(X,Y,Z, cmap='hot', aspect='auto', origin='lower')

            pdf.savefig()
            plt.close(fig)


@click.command()
@click.option("--data-path", default="../data/")
@click.option("--results-path", default="../results")
@click.option("--place-field-posterior", is_flag=True)
@click.option("--theta-delta-t-ms", default=10)
@click.option("--theta-time-step-ms", default=5)
@click.option("--replay-delta-t-ms", default=3)
@click.option("--replay-time-step-ms", default=3)
@click.option("--velocity-cutoff", default=10)
@click.option("--model", default="map", type=click.Choice(['map', 'momentum', 'gridsearch']))
@click.option("--bin-size-cm", default=2)
@click.option("--rerun", is_flag=True)
@click.option("--skip-linear", is_flag=True)
@click.option("--rats", multiple=True, type=click.Choice(hsep.RAT_NAMES), default=hsep.RAT_NAMES)
@click.option("--checkpoint-path", default="../checkpoints/")
@click.option("--approximation-method", type=click.Choice(['iterative', 'analytic']), default='analytic')
@click.option("--normalize", is_flag=True)
def main(
        data_path, 
        results_path, 
        place_field_posterior,
        theta_delta_t_ms,
        theta_time_step_ms,
        replay_delta_t_ms,
        replay_time_step_ms,
        velocity_cutoff,
        model,
        bin_size_cm,
        rerun,
        skip_linear,
        rats,
        checkpoint_path,
        approximation_method, 
        normalize
    ):
    print(f"Processing rats: {rats}", file=sys.stdout)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    to_dump = locals()
    with open(os.path.join(results_path, "params.json"), 'w') as f:
        f.write(json.dumps(to_dump, indent=4, sort_keys=True))
        print(json.dumps(to_dump))

    for rat in rats:
        rat_data = os.path.join(data_path, rat)
        for session in os.listdir(rat_data):
            results = os.path.join(results_path, rat, session)
            if os.path.exists(results) and not rerun:
                print(f"Skipping {rat} {session} {model}", file=sys.stdout)
                continue
            track_type = session[:-1]
            if skip_linear and track_type == "Linear":
                continue
            os.makedirs(results, exist_ok=True)

            try: 
                env_size = None if track_type == "Linear" else (0,0,200,200)
                (
                    raw_data,
                    place_field_data,
                    theta_data,
                    ripple_data
                ) = hsep.preprocess_data(
                    rat_name                      = rat,
                    session                       = int(session[-1]),
                    data_path                     = data_path,
                    track_type                    = track_type,
                    velocity_cutoff               = velocity_cutoff,
                    theta_time_window_ms          = theta_delta_t_ms,
                    theta_time_window_advance_ms  = theta_time_step_ms,
                    ripple_time_window_ms         = replay_delta_t_ms,
                    ripple_time_window_advance_ms = replay_time_step_ms,
                    environment_size              = env_size,
                    bin_size_cm                   = bin_size_cm
                )
                print(f"Finished pre-processing {rat} {session}", file=sys.stdout)

                if track_type == 'Linear':
                    env_size = (
                        np.min(raw_data.raw_position['x']),
                        np.min(raw_data.raw_position['y']),
                        np.max(raw_data.raw_position['x']),
                        np.max(raw_data.raw_position['y'])
                    )

                plot_place_fields(
                    place_field_data,
                    raw_data,
                    track_type,
                    results
                )
                try:
                    ckpt = os.path.join(checkpoint_path, rat, session, model, "theta")
                    if os.path.exists(ckpt):
                        print(f"Loading checkpoint {ckpt}", file=sys.stdout)
                        theta_model,theta_values = hsem.resume_from_checkpoint(
                            ckpt
                        )
                    else:
                        theta_model,theta_values = run_model(
                            model_selection  = model,
                            place_field_data = place_field_data,
                            dt               = theta_delta_t_ms / 1000,
                            bin_size         = bin_size_cm,
                            spikemats        = theta_data.theta_spikes,
                            environment_size = env_size,
                            checkpoint_path  = ckpt,
                            approximation_method = approximation_method, 
                            normalize        = normalize
                        )
                    hseu.save_pickle(theta_model, os.path.join(results, "theta_model.pkl"))
                    hseu.save_pickle(theta_values, os.path.join(results, "theta_values.pkl"))

                    print(f"Finished theta {rat} {session}", file=sys.stdout)

                    plot_theta(
                        theta_values,
                        theta_data.true_trajectory,
                        track_type,
                        results
                    )
                    plot_model_approximations(
                        theta_model,
                        results
                    )
                except Exception as e:
                    print(f"Failed to process {rat}:{session} theta. Skipping...", file=sys.stderr)
                    print(e, file=sys.stderr)

                try: 
                    ckpt = os.path.join(checkpoint_path, rat, session, model, "ripple")
                    if os.path.exists(ckpt):
                        print(f"Loading checkpoint {ckpt}", file=sys.stdout)
                        ripple_model,ripple_values = hsem.resume_from_checkpoint(
                            ckpt
                        )
                    else:
                        ripple_model,ripple_values = run_model(
                            model_selection  = model,
                            place_field_data = place_field_data,
                            dt               = replay_delta_t_ms / 1000,
                            bin_size         = bin_size_cm,
                            spikemats        = ripple_data.ripple_spikes,
                            environment_size = env_size,
                            checkpoint_path  = ckpt,
                            approximation_method = approximation_method,
                            normalize        = normalize
                        )
                    hseu.save_pickle(ripple_model, os.path.join(results, "ripple_model.pkl"))
                    hseu.save_pickle(ripple_values, os.path.join(results, "ripple_values.pkl"))

                    print(f"Finished replay {rat} {session}", file=sys.stdout)

                    plot_replay(
                        ripple_values,
                        results
                    )
                    plot_model_approximations(
                        ripple_model,
                        results
                    )
                except Exception as e:
                    print(f"Failed to process {rat}:{session} ripple. Skipping...", file=sys.stderr)
                    print(e, file=sys.stderr)

                print(f"Rat {rat} session {session} complete.", file=sys.stdout)

            except Exception as e:
                print(f"Failed to process {rat}:{session}. Skipping...", file=sys.stderr)
                print(e, file=sys.stderr)

if __name__ == '__main__':
    main()