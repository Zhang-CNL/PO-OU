import click
import os
import sys 
import json
import traceback
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("."))

from pathlib import Path
from typing import Any

import hippocampalseq as hse
import hippocampalseq.io as hseio
import hippocampalseq.utils as hseu
import hippocampalseq.plotting as hsepl
import hippocampalseq.models as hsem

MODELS = {
    "MAP"                  : hsem.BayesianMAP,
    "Momentum"             : hsem.Momentum,
    "MomentumVelocity"     : hsem.MomentumVelocity,
    "MomentumVelocityBias" : hsem.MomentumVelocityBias,
    "CANNDynamics"         : hsem.CANNDynamics,
    "CANNDynamicsBias"     : hsem.CANNDynamicsSpikes
}

def run_model(
        model_name: str,
        model_kwargs: dict[str, Any],
        model_fit_kwargs: dict[str, Any] = {},
        model_pred_kwargs: dict[str, Any] = {}
    ):
    try:
        model = MODELS[model_name](**model_kwargs)
        decoded = model.fit(**model_fit_kwargs)
        if isinstance(decoded, tuple):
            out = {
                'training'   : decoded[0],
                'validation' : decoded[1]
            }
        else:
            out = {
                'training': decoded
            }
        if len(model_pred_kwargs) > 0:
            predicted = model.transform(**model_pred_kwargs)
            out['testing'] = predicted
        return model, out
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print('-'*100, file=sys.stderr)
        print(f"Model {model_name} failed")
        return None,{}

def run_theta_models(
        raw_data: hse.RawData, 
        place_field_data: hse.PlaceFields, 
        theta_data: hse.Theta,
        parameters: dict[str, Any],
        results_path: str|Path
    ):
    place_fields = place_field_data.place_fields[place_fields.place_cell_ids]
    environment_size = raw_data.environment_size

    indices = hseu.train_test_valid_split(
        len(theta_data.spikes), 
        parameters.get('percent_train', 0.8),
        parameters.get('percent_validation', 0.1)
    )

    true_velocity = [
        gt[[f'V_{d}' for d in environment_size.axes()]].values
        for gt in theta_data.ground_truth
    ]
    true_position = [
        gt[environment_size.axes()].values
        for gt in theta_data.ground_truth
    ]
    extract_indices = lambda ls,idx: [ls[i] for i in idx] if len(idx) > 0 else None
    tv_train,tv_valid,tv_test = [
        extract_indices(true_velocity, idx) 
        for idx in indices
    ]
    tp_train,tp_valid,tp_test = [
        extract_indices(true_position, idx)
        for idx in indices
    ]
    sp_train,sp_valid,sp_test = [
        extract_indices(theta_data.spikes, idx)
        for idx in indices
    ]

    output = {}

    map_model,map_results = run_model(
        "MAP",
        {
            'place_fields' : place_fields,
            'dt'           : parameters.get('time_window_s'),
            'bin_size'     : parameters.get('bin_size', 2.0)
        },
        {
            'X' : theta_data.spikemat
        }
    )
    output |= {
        "MAP": map_results
    }

    momentum_model,momentum_results = run_model(
        "Momentum", 
        {
            'dt'               : parameters.get('time_window_s'),
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
        },
        {},
        {
            'spikemats' : sp_test
        }
    )
    output |= {
        "Momentum": momentum_results
    }

    momentum_vel_model,momentum_vel_results = run_model(
        "MomentumVelocity",
        {
            'dt'               : parameters.get('time_window_s'),
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
            'velocity_type'    : 'true' 
        },
        {
            'Xtrain' : tv_train,
            'Xvalid' : tv_valid
        },
        {
            'spikemats' : sp_test,
            'Xtest'     : tv_test
        }
    )
    output |= {
        "MomentumVelocityTrue": momentum_vel_results
    }

    momentum_vellik_model,momentum_vellik_results = run_model(
        "MomentumVelocity",
        {
            'dt'               : parameters.get('time_window_s'),
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
            'velocity_type'    : 'observed'
        },
        {
        },
        {
            'spikemats' : sp_test
        }
    )
    output |= {
        "MomentumVelocityLikelihood": momentum_vellik_results
    }

    momentum_vbias_model,momentum_vbias_results = run_model(
        "MomentumVelocityBias",
        {
            'velocity_train'   : tv_train,
            'velocity_valid'   : tv_valid,
            'bias_fn'          : 'linear',
            'dt'               : parameters.get('time_window_s'),
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
        },
        {
        },
        {
            'spikemats' : sp_test,
            'Xtest'     : tv_test
        }
    )
    output |= {
        "MomentumVelocityBias": momentum_vbias_results
    }

    cann_model,cann_results = run_model(
        "CANNDynamics",
        {
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'dt'                  : theta_time_window_s,
            'bin_size'            : bin_size,
            'environment_size'    : raw_data.environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
        },
        {
        },
        {
            'spikemats' : sp_test,
            'Xtest'     : tp_test
        }
    )
    output |= {
        "CANNDynamics": cann_results
    }

    cann_bias_model,cann_bias_results = run_model(
        "CANNDynamicsBias",
        {
            'spikemat'         : sp_train,
            'place_fields'     : place_fields,
            'dt'               : parameters.get('time_window_s'),
            'bin_size'         : parameters.get('bin_size', 2.0),
            'environment_size' : environment_size,
            'true_position'    : tp_train,
        }, {}, {
            'spikemat': sp_test,
            'true_position': tp_test
        }
    )
    output |= {
        "CANNDynamicsBias": cann_bias_results
    }

    hseio.save_to_mat2(
        results_path / "model_theta_results.mat",
        {
            "split_indices" : {
                "train"      : indices[0],
                "validation" : indices[1],
                "test"       : indices[2]
            } 
        } | output
    )

@click.command()
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rats", multiple=True, type=click.Choice(hseio.RAT_NAMES), default=hseio.RAT_NAMES)
def main(
        results_path: str,
        run_config: str,
        rats: list[str],
    ):
    if not isinstance(rats, list):
        rats = [rats]
    results_path = Path(results_path)
    run_config = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())
    
    results_path = results_path / parameters.get("name")
    print(results_path)

    for rat in rats:
        rat_path = results_path / rat
        for session in os.listdir(rat_path):
            results_dir = rat_path / session
            track_type = session[:-1]
            session_n  = int(session[-1])

            print(f"Training models on {rat}:{session}")
            raw_data = hseio.load_from_mat2(results_dir / "raw_data.mat")
            place_field_data = raw_data['place_field_data']
            raw_data = raw_data['raw_data']

            if not paramaters.get("ignore_theta", False):
                theta_data = hseo.load_from_mat2(results_dir / "theta_data.mat")
                run_theta_models(
                    raw_data,
                    place_field_data,
                    theta_data,
                    results_dir
                )

            if not parameters.get("ignore_ripple", False):
                raise NotImplementedError

if __name__ == "__main__":
    main()