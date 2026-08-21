import click
import os 
import sys 
import json 
sys.path.append(os.path.realpath("..")) # Add hippocampalseq to path
sys.path.append(os.path.realpath(".")) # Add job utils to path

import hippocampalseq.io as hseio

import analyze

@click.command()
@click.option("--results-path", default="../results/")
@click.option("--run-config")
@click.option("--profile-name", default=None)
@click.option("--rats", multiple=True, type=click.Choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        results_path: str,
        run_config: str,
        profile_name: str,
        rats: list[str],
    ):
    with open(os.path.realpath(run_config), 'r') as f:
        parameters = json.loads(f.read())

    profile_name = parameters.get("name", profile_name)
    print(profile_name)
    results_path = os.path.join(results_path, profile_name)

    for rat in rats:
        rat_path = os.path.join(results_path, rat)
        for session in os.listdir(rat_path):
            results_dir = os.path.join(rat_path, session)
            track_type = session[:-1]
            session_n  = int(session[-1])
            print(f"Analyzing {rat} {track_type} {session_n}")

            raw_data = hseio.load_from_mat2(
                os.path.join(results_dir, 'raw_data.mat')
            )

            analyze.raw_analysis(
                results_dir,
                track_type,
                raw_data
            )

            theta_path = os.path.join(results_dir, 'theta_raw.mat')
            model_results_path = os.path.join(results_dir, 'model_theta_results.mat')
            if os.path.exists(model_results_path) and os.path.exists(theta_path):
                model_results = hseio.load_from_mat2(model_results_path)
                theta_data = hseio.load_from_mat2(theta_path)
                analyze.theta_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    theta_data,
                    model_results,
                    parameters
                )

            replay_path = os.path.join(results_dir, 'replay_data.mat')
            model_results_path = os.path.join(results_dir, 'model_swr_results.mat')
            if os.path.exists(model_results_path) and os.path.exists(replay_path):
                model_results = hseio.load_from_mat2(model_results_path)
                replay_data = hseio.load_from_mat2(replay_path)
                analyze.replay_analysis(
                    results_dir,
                    track_type,
                    raw_data,
                    replay_data,
                    parameters
                )
                
    print("Job completed")


if __name__ == '__main__':
    main()