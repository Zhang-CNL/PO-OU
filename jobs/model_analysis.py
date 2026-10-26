import click
import os
import sys
import json
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))

from pathlib import Path

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.utils as hseu
import hippocampalseq.plotting as hsepl
import hippocampalseq.models as hsem

def theta_model_analysis(
        theta_model_data
    ):
    pass

@click.command()
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rats", multiple=True, type=click.choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        results_path: str,
        run_config: str,
        rats: list[str]
    ):
    if not isinstance(rats, list):
        rats = [rats]
    results_path = Path(results_path)
    run_config = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())

    results_path /= parameters["name"]
    print(results_path)

    for rat in rats:
        rat_path = results_path / rat 
        for session in os.listdir(rat_path):
            results_dir = rat_path / session
            track_type = session[:-1]
            session_n = int(session[-1])

            print(f"Analysis for {rat}:{session}")

            if not parameters.get("ignore_theta", False):
                theta_model_data = hseio.load_from_mat2(
                    results_dir / "model_theta_results.mat"
                )
                theta_model_analysis(
                    theta_model_data
                )

            if not parameters.get("ignore_ripple", False):
                raise NotImplementedError

if __name__ == '__main__':
    main()