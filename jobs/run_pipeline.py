import os
import sys 
import click
sys.path.append(os.path.realpath(".."))

from tqdm import tqdm
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
        environment_size: tuple
    ):
    if model_selection == "map":
        model = hsem.BayesianMAP(place_field_data.place_fields, dt, bin_size)
    elif model_selection == "momentum":
        model = hsem.Momentum(place_field_data.place_fields, spikemats, dt, environment_size, bin_size)
    elif model_selection == "gridsearch":
        pass

def plot_place_fields(
        place_field_data: hsep.PlaceFields, 
        raw_data: nap.TsdFrame, 
        track_type: str, 
        results_path: str 
    ):
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
        plt.savefig(os.path.join(results_path, f"place_field_{i}.svg"), dpi=300)
        plt.close()



@click.command()
@click.option("--data-dir", default="../data/")
@click.option("--results-dir", default="../results")
@click.option("--place-field-posterior", default=False)
@click.option("--theta-delta-t-ms", default=10)
@click.option("--theta-time-step-ms", default=5)
@click.option("--replay-delta-t-ms", default=5)
@click.option("--replay-time-step-ms", default=5)
@click.option("--velocity-cutoff", default=10)
@click.option("--model", default="map", type=click.Choice(['map', 'momentum', 'gridsearch']))
def main(
        data_dir, 
        results_dir, 
        place_field_posterior,
        theta_delta_t_ms,
        theta_time_step_ms,
        replay_delta_t_ms,
        replay_time_step_ms,
        velocity_cutoff,
        model,
    ):
    for rat in tqdm(hsep.RAT_NAMES):
        rat_data = os.path.join(data_dir, rat)
        for session in os.listdir(rat_data):
            results = os.path.join(results_dir, rat, session)
            if not os.path.exists(results):
                os.makedirs(results)
            track_type = session[:-1]

            try: 
                (
                    raw_data,
                    place_field_data,
                    theta_data,
                    ripple_data
                ) = hsep.preprocess_data(
                    rat_name                      = rat,
                    session                       = int(session[-1]),
                    data_path                     = data_dir,
                    track_type                    = track_type,
                    velocity_cutoff               = velocity_cutoff,
                    theta_time_window_ms          = theta_delta_t_ms,
                    theta_time_window_advance_ms  = theta_time_step_ms,
                    ripple_time_window_ms         = replay_delta_t_ms,
                    ripple_time_window_advance_ms = replay_time_step_ms,
                    environment_size              = None if track_type == "Linear" else (0,0,200,200)
                )

                plot_place_fields(
                    place_field_data,
                    raw_data,
                    track_type,
                    results
                )
            except Exception as e:
                print(f"Failed to process {rat}:{session}. Skipping...")
                print(e)

if __name__ == '__main__':
    main()