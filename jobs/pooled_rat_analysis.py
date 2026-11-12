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

def pool_across_rats(rat_sweeps: list[list[dict]], track_type: str, results_path: Path):
    def norm(x):
        col = x.sum(axis=0, keepdims=True)
        return x / np.where(col==0, 1.0, col)

    for name in rat_sweeps[0][0]:
        all_model_sessions = []
        for session in rat_sweeps:
            all_model_sessions += [sess[name]['dist_of_peak_post'] for sess in session]
        stack = np.stack(all_model_sessions, axis=-1) 
        average_posterior = np.mean(
            [norm(stack[...,i]) for i in range(stack.shape[-1])],
            axis=0
        )
        with PdfPages(results_path / f"{name}_{track_type}_theta_sweeps.pdf") as pdf:
            hsepl.plot_across_session_decoding_hist(
                hsea.max_probability([average_posterior])[0],
                average_posterior
            )
            pdf.savefig()
            plt.close()

@click.command()
@click.option("--results-path", default="../results")
@click.option("--run-config")
def main(results_path: str, run_config: str):
    results_path = Path(results_path)
    run_config   = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())
    
    results_path = results_path / parameters['name'] 

    theta_linear_sweeps = []
    theta_open_sweeps   = []

    for rat in os.listdir(results_path):
        ratdir = results_path / rat
        if not ratdir.is_dir():
            continue
        if not parameters.get("ignore_theta", False):
            linear_file = ratdir / "linear_theta.pkl.bz"
            open_file   = ratdir / "open_theta.pkl.bz"
            if linear_file.exists():
                theta_linear_sweeps.append(compress_pickle.load(linear_file))
            if open_file.exists():
                theta_open_sweeps.append(compress_pickle.load(open_file))

    if len(theta_linear_sweeps) > 0:
        pool_across_rats(
            theta_linear_sweeps,
            "Linear",
            results_path
        )
    if len(theta_open_sweeps) > 0:
        pool_across_rats(
            theta_open_sweeps,
            "Open",
            results_path
        )

if __name__ == '__main__':
    main()
