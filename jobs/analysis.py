import click
import os 
import sys 
import json 
sys.path.append(os.path.realpath("..")) # Add hippocampalseq to path

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.plotting as hsepl
import hippocampalseq.utils as hseu

def raw_analysis(results_path, track_type, raw_data):
    hsepl.plot_place_fields(
        hseu.AttrDict(raw_data['place_field_data']),
        hseu.AttrDict(raw_data['raw_data']),
        track_type,
        results_path
    ) 
    with PdfPages(os.path.join(results_path, "true_trajectories.pdf")) as pdf:
        hsepl.plot_trajectories(
            raw_data['theta_data']['true_trajectories']
        )

def theta_analysis(results_path, track_type, raw_data, theta_data):
    true_trajectories = raw_data['theta_data']['true_trajectories']

    with PdfPages(os.path.join(results_path, "theta.pdf")) as pdf:
        pass


def replay_analysis(results_path, track_type, raw_data, replay_data):
    pass

@click.command()
@click.option("--results-path", default="../results/")
@click.option("--run-config")
@click.option("--rats", multiple=True, type=click.Choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        results_path,
        run_config,
        rats,
    ):
    with open(os.path.realpath(run_config), 'r') as f:
        parameters = json.loads(f)

    for rat in rats:
        rat_path = os.path.join(results_path, rat)
        for session in os.listdir(rat_path):
            results_dir = os.path.join(rat_path, session)
            track_type = session[:-1]
            session_n  = int(session[-1])

            raw_data = sio.loadmat(
                os.path.join(results_dir, 'raw_data.mat')
            )
            raw_analysis(
                results_dir,
                track_type,
                raw_data
            )

            theta_path = os.path.join(results_dir, 'model_theta_results.mat')
            if os.path.exists(theta_path):
                theta_data = sio.loadmat(theta_path)
                theta_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    theta_data,
                )

            replay_path = os.path.join(results_dir, 'model_swr_results.mat')
            if os.path.exists(replay_path):
                replay_data = sio.loadmat(replay_path)
                replay_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    replay_data,
                )
                



if __name__ == '__main__':
    main()