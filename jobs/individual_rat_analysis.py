import click
import os
import sys
import json
import compress_pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.analysis as hsea
import hippocampalseq.utils as hseu
import hippocampalseq.plotting as hsepl
import hippocampalseq.models as hsem

def plot_trajectory(
            true_trajectory, 
            model_trajectory, 
            cumulative_probability, 
            environment_size,
            model_name
        ):
    fig,ax = plt.subplots(1,2,figsize=(20,10),dpi=300,constrained_layout=True)
    hsepl.plot_trajectories({
        'True Trajectory': hseu.atleast_2d(true_trajectory),
        model_name : hseu.atleast_2d(model_trajectory)
    }, environment_size=environment_size, ax=ax[0])

    if len(environment_size) == 2:
        xlen = environment_size.x[1] - environment_size.x[0]
        ylen = environment_size.y[1] - environment_size.y[0]
        if xlen == ylen:
            ax[0].set_aspect('equal')
    if cumulative_probability.shape[-1] > 1:
        cumulative_probability = cumulative_probability.T

    im = ax[1].imshow(
        cumulative_probability, 
        origin='lower', 
        cmap='hot_r',
    )

    if len(environment_size) == 2:
        xlen = environment_size.x[1] - environment_size.x[0]
        ylen = environment_size.y[1] - environment_size.y[0]
        if xlen == ylen:
            ax[1].set_aspect('equal')

    fig.colorbar(im, ax=ax[1])
    return fig

def theta_model_analysis(
        raw_data,
        theta_model_data,
        parameters,
        results_path
    ):
    train_test_idx = theta_model_data['split_indices']
    del theta_model_data['split_indices']

    theta_data = raw_data['theta_data']
    environment_size = raw_data['raw_data'].environment_size
    bin_size = parameters.get('bin_size', 2)

    analysis_out = {}

    aics = []
    bics = []
    loglikes = []
    model_names = []

    ground_truth_positions = [
        gt[environment_size.axes()].values
        for gt in theta_data.ground_truth
    ]

    for model_name,model_data in theta_model_data.items():
        print(f"Theta {model_name} plotting started")
        if model_name == "MAP":
            spkct_percent = 66.67
            true_position = theta_data.ground_truth
            spikes = theta_data.spikes
            normalized_posterior = hsea.bayesian_map_to_grids(
                model_data['results']['training'].posterior_probabilities
            )
        else:
            spkct_percent = None
            true_position = [
                theta_data.ground_truth[i] for i in train_test_idx['train']
            ] + [
                theta_data.ground_truth[i] for i in train_test_idx['validation']
            ]
            spikes = [
                theta_data.spikes[i] for i in train_test_idx['train']
            ] + [
                theta_data.spikes[i] for i in train_test_idx['validation']
            ]
            normalized_posterior = hsea.gaussians_to_grids(
                model_data['results']['training'].smoothed_mean \
                    + model_data['results']['validation'].smoothed_mean,
                model_data['results']['training'].smoothed_cov \
                    + model_data['results']['validation'].smoothed_cov,
                environment_size,
                bin_size,
                model_data['model']['marginal_position_dims']
            )

        if len(environment_size) == 1:
            peak_posterior = hsea.peak_posterior(
                normalized_posterior,
                environment_size,
                true_position,
                spikes,
                bin_size_cm=bin_size,
                spike_count_percentile=spkct_percent
            )
        else:
            peak_posterior = hsea.peak_posterior_2d(
                normalized_posterior,
                environment_size,
                true_position,
                spikes,
                bin_size_cm=bin_size,
                spike_count_percentile=spkct_percent
            )

        with PdfPages(results_path / f"{model_name}_theta_trajectories.pdf") as pdf:
            if model_name == "MAP":
                model_trajectories = model_data['results']['training'].decoded_trajectories
                model_cumprobs = model_data['results']['training'].cumulative_probabilities
                true_position = ground_truth_positions
            else:
                model_trajectories = model_data['results']['training'].smoothed_mean
                model_cumprobs = model_data['results']['training'].cumulative_probabilities
                true_position = [ground_truth_positions[i] for i in train_test_idx['train']]
                model_trajectories = [
                    mt[:,model_data['model']['marginal_position_dims']]
                    for mt in model_trajectories
                ]

            for i in range(len(model_trajectories)):
                fig = plot_trajectory(
                    true_position[i],
                    model_trajectories[i],
                    model_cumprobs[i],
                    environment_size,
                    model_name
                )
                pdf.savefig(fig)
                plt.close(fig)

            if 'validation' in model_data['results']:
                model_trajectories = model_data['results']['validation'].smoothed_mean
                model_cumprobs = model_data['results']['validation'].cumulative_probabilities
                true_position = [ground_truth_positions[i] for i in train_test_idx['validation']]
                model_trajectories = [
                    mt[:,model_data['model']['marginal_position_dims']]
                    for mt in model_trajectories
                ]
                for i in range(len(model_trajectories)):
                    fig = plot_trajectory(
                        true_position[i],
                        model_trajectories[i],
                        model_cumprobs[i],
                        environment_size,
                        model_name
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

        if model_name != "MAP":
            with PdfPages(results_path / f"{model_name}_theta_fit_curves.pdf") as pdf:
                plt.figure(dpi=300)
                plt.subplot(1,2,1)
                plt.title("Training negative log-likelihood")
                plt.plot(model_data['results']['training'].loglike)
                plt.subplot(1,2,2)
                plt.title("Validation negative log-likelihood")
                plt.plot(model_data['results']['validation'].loglike)
                pdf.savefig()
                plt.close()

            aics.append(model_data['results']['training'].aic)
            bics.append(model_data['results']['training'].bic)
            loglikes.append(model_data['results']['training'].loglike_full)
            model_names.append(model_name)

        analysis_out[model_name] = peak_posterior

    with PdfPages(results_path / f"model_comparison.pdf") as pdf:
        plt.figure(dpi=300)
        plt.subplot(1,3,1)
        plt.bar(model_names, aics)
        plt.title("AIC")
        plt.xticks(rotation=45)

        plt.subplot(1,3,2)
        plt.bar(model_names, bics)
        plt.title("AIC")
        plt.xticks(rotation=45)

        plt.subplot(1,3,3)
        plt.bar(model_names, loglikes)
        plt.title("Log-likelihood")
        plt.xticks(rotation=45)
        pdf.savefig()
        plt.close()

    return analysis_out

def theta_session_analysis(session_data, track_type, results_path):
    def norm(x):
        col = x.sum(axis=0, keepdims=True)
        return x / np.where(col == 0, 1.0, col)

    for name in session_data[0]:
        model_data = [session[name] for session in session_data]

        rat_pooled = np.stack(
            [data["dist_of_peak_post"] for data in model_data],
            axis=-1,
        )
        rat_pooled = np.mean(
            [norm(rat_pooled[..., i]) for i in range(rat_pooled.shape[-1])],
            axis=0,
        )

        with PdfPages(results_path / f"{name}_{track_type}_theta_sweeps.pdf") as pdf:
            hsepl.plot_across_session_decoding_hist(
                hsea.max_probability([rat_pooled])[0],
                rat_pooled,
            )
            pdf.savefig()

def replay_model_analysis(
        raw_data,
        replay_model_data,
        parameters
    ):
    pass

@click.command()
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rat", type=click.Choice(hseio.RAT_NAMES))
def main(results_path: str, run_config: str, rat: str):
    results_path = Path(results_path)
    run_config   = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())

    results_path = results_path / parameters["name"] / rat

    linear_tracks = []
    open_tracks = []

    for session in os.listdir(results_path):
        results_dir = results_path / session
        if not results_dir.is_dir():
            continue
        track_type = session[:-1]
        session_n = int(session[-1])

        print(f"Analysis for {rat}:{session}")

        raw_data = compress_pickle.load(
            results_dir / "preprocessed_data.pkl.bz"
        )

        if not parameters.get("ignore_theta", False):
            theta_model_data = compress_pickle.load(
                results_dir / "model_theta_results.pkl.bz"
            )
            session_data = theta_model_analysis(
                raw_data,
                theta_model_data,
                parameters,
                results_dir
            )
            if track_type == "Linear":
                linear_tracks.append(session_data)
            else:
                open_tracks.append(session_data)

        if not parameters.get("ignore_ripples", False):
            replay_model_data = compress_pickle.load(
                results_dir / "model_replay_results.pkl.bz"
            )
            replay_model_analysis(
                raw_data,
                replay_model_data,
                parameters
            )

    if len(linear_tracks) > 0:
        compress_pickle.dump(
            linear_tracks,
            results_path / "linear_theta.pkl.bz"
        )
        theta_session_analysis(
            linear_tracks,
            "Linear",
            results_path
        )

    if len(open_tracks) > 0:
        compress_pickle.dump(
            open_tracks,
            results_path / "open_theta.pkl.bz"
        )
        theta_session_analysis(
            open_tracks, 
            "Open",
            results_path
        )

if __name__ == '__main__':
    main()