import click
import os
import sys 
import json
import traceback
import compress_pickle
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
    "Gaussian"             : hsem.GaussianApproximation,
    "Diffusion"            : hsem.spiking.Diffusion,
    "DiffusionScaled"      : hsem.spiking.DiffusionScaled,
    "Momentum"             : hsem.spiking.Momentum,
    "MomentumVelocity"     : hsem.spiking.MomentumVelocity,
    "MomentumVelocityBias" : hsem.spiking.MomentumVelocityBias,
    "CANNDynamics"         : hsem.spiking.CANNDynamics,
    "CANNDynamicsBias"     : hsem.spiking.CANNDynamicsSpikes,
    "OUDynamics"           : hsem.spiking.OUDynamics,
    "OUDynamicsScaled"     : hsem.spiking.OUDynamicsScaled
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
    place_fields = place_field_data.place_fields[place_field_data.place_cell_ids]
    environment_size = raw_data.environment_size

    indices = hseu.train_test_valid_split(
        len(theta_data.spikes), 
        parameters.get('percent_train', 0.8),
        parameters.get('percent_validation', 0.2)
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
    tv_train,tv_valid = [
        extract_indices(true_velocity, idx) 
        for idx in indices
    ]
    tp_train,tp_valid = [
        extract_indices(true_position, idx)
        for idx in indices
    ]
    sp_train,sp_valid = [
        extract_indices(theta_data.spikes, idx)
        for idx in indices
    ]

    output = {}

    map_model,map_results = run_model(
        "MAP",
        {
            'place_fields' : place_fields,
            'dt'           : parameters['theta_args']['time_window_s'],
            'bin_size'     : parameters.get('bin_size', 2.0)
        },
        {
            'X' : theta_data.spikes
        }
    )
    output |= {
        "MAP": {
            'model'   : map_model,
            'results' : map_results
        }
    }

    momentum_model,momentum_results = run_model(
        "Momentum", 
        {
            'dt'               : parameters['theta_args']['time_window_s'],
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
        },
    )
    output |= {
        "Momentum": {
            'model'   : momentum_model,
            'results' : momentum_results
        }
    }

    momentum_vel_model,momentum_vel_results = run_model(
        "MomentumVelocity",
        {
            'dt'               : parameters['theta_args']['time_window_s'],
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
    )
    output |= {
        "MomentumVelocityTrue": {
            'model'   : momentum_vel_model,
            'results' : momentum_vel_results
        }
    }

    momentum_vellik_model,momentum_vellik_results = run_model(
        "MomentumVelocity",
        {
            'dt'               : parameters['theta_args']['time_window_s'],
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
            'velocity_type'    : 'observed'
        },
    )
    output |= {
        "MomentumVelocityLikelihood": {
            'model'   : momentum_vellik_model,
            'results' : momentum_vellik_results
        }
    }

    momentum_vbias_model,momentum_vbias_results = run_model(
        "MomentumVelocityBias",
        {
            'velocity_train'   : tv_train,
            'velocity_valid'   : tv_valid,
            'bias_fn'          : 'linear',
            'dt'               : parameters['theta_args']['time_window_s'],
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
        },
    )
    output |= {
        "MomentumVelocityBias": {
            'model'   : momentum_vbias_model,
            'results' : momentum_vbias_results
        }
    }

    cann_model,cann_results = run_model(
        "CANNDynamics",
        {
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'dt'                  : parameters['theta_args']['time_window_s'],
            'bin_size'            : parameters.get('bin_size', 2.0),
            'environment_size'    : environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
        },
    )
    output |= {
        "CANNDynamics": {
            'model'   : cann_model,
            'results' : cann_results
        }
    }

    cann_bias_model,cann_bias_results = run_model(
        "CANNDynamicsBias",
        {
            'true_position_train'   : tp_train,
            'true_position_valid'   : tp_valid,
            'dt'                    : parameters['theta_args']['time_window_s'],
            'bin_size'              : parameters.get('bin_size', 2.0),
            'environment_size'      : environment_size,
            'place_fields'          : place_fields,
            'spikemat_train'        : sp_train,
            'spikemat_valid'        : sp_valid,
            'initialization_method' : 'uniform'
        }
    )
    output |= {
        "CANNDynamicsBias": {
            'model'   : cann_bias_model,
            'results' : cann_bias_results
        }
    }

    ou_model,ou_results = run_model(
        "OUDynamics",
        {
            'dt'                  : parameters['theta_args']['time_window_s'],
            'bin_size'            : parameters.get('bin_size', 2.0),
            'environment_size'    : environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'bias'                : True
        }, 
        {
        },
        {
            'spikemats' : sp_test,
            'X'         : tp_test
        }
    )
    output |= {
        "OUDynamicsBias": {
            'model'   : ou_model,
            'results' : ou_results
        }
    }

    ounb_model,ounb_results = run_model(
        "OUDynamics",
        {
            'dt'                  : parameters['theta_args']['time_window_s'],
            'bin_size'            : parameters.get('bin_size', 2.0),
            'environment_size'    : environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'bias'                : False
        }, 
        {
        },
        {
            'spikemats' : sp_test,
            'X'         : tp_test
        }
    )
    output |= {
        "OUDynamicsNoBias": {
            'model'   : oub_model,
            'results' : oub_results
        }
    }

    ous_model,ous_results = run_model(
        "OUDynamicsScaled",
        {
            'dt'                  : parameters['theta_args']['time_window_s'],
            'bin_size'            : parameters.get('bin_size', 2.0),
            'environment_size'    : environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'bias'                : False,
            'initialization_method': 'uniform'
        }, 
    )
    output |= {
        "OUDynamicsScaled": {
            'model'   : ous_model,
            'results' : ous_results
        }
    }

    ousnb_model,ousnb_results = run_model(
        "OUDynamicsScaled",
        {
            'dt'                  : parameters['theta_args']['time_window_s'],
            'bin_size'            : parameters.get('bin_size', 2.0),
            'environment_size'    : environment_size,
            'place_fields'        : place_fields,
            'spikemat_train'      : sp_train,
            'spikemat_valid'      : sp_valid,
            'true_position_train' : tp_train,
            'true_position_valid' : tp_valid,
            'bias'                : False,
            'initialization_method': 'uniform'
        }, 
    )
    output |= {
        "OUDynamicsScaledNoBias": {
            'model'   : ousnb_model,
            'results' : ousnb_results
        }
    }

    compress_pickle.dump(
        {
            "split_indices" : {
                "train"      : indices[0],
                "validation" : indices[1],
            } 
        } | output,
        results_path / "model_theta_results.mat",
    )
    
def run_replay_models(
        raw_data: hse.RawData, 
        place_field_data: hse.PlaceFields, 
        replay_data: hse.Replay,
        parameters: dict[str, Any],
        results_path: str|Path
    ):
    place_fields = place_field_data.place_fields[place_field_data.place_cell_ids]
    environment_size = raw_data.environment_size

    indices = hseu.train_test_valid_split(
        len(theta_data.spikes), 
        parameters.get('percent_train', 0.8),
        parameters.get('percent_validation', 0.2)
    )

    true_velocity = [
        gt[[f'V_{d}' for d in environment_size.axes()]].values
        for gt in replay_data.ground_truth
    ]
    true_position = [
        gt[environment_size.axes()].values
        for gt in replay_data.ground_truth
    ]
    extract_indices = lambda ls,idx: [ls[i] for i in idx] if len(idx) > 0 else None
    tv_train,tv_valid = [
        extract_indices(true_velocity, idx) 
        for idx in indices
    ]
    tp_train,tp_valid = [
        extract_indices(true_position, idx)
        for idx in indices
    ]
    sp_train,sp_valid = [
        extract_indices(theta_data.spikes, idx)
        for idx in indices
    ]

    output = {}

    map_model,map_results = run_model(
        "MAP",
        {
            'place_fields' : place_fields,
            'dt'           : parameters['theta_args']['time_window_s'],
            'bin_size'     : parameters.get('bin_size', 2.0)
        },
        {
            'X' : theta_data.spikes
        }
    )
    output |= {
        "MAP": {
            'model'   : map_model,
            'results' : map_results
        }
    }

    momentum_model,momentum_results = run_model(
        "Momentum", 
        {
            'dt'               : parameters['theta_args']['time_window_s'],
            'environment_size' : environment_size,
            'bin_size'         : parameters.get('bin_size', 2.0),
            'place_fields'     : place_fields,
            'spikemat_train'   : sp_train,
            'spikemat_valid'   : sp_valid,
        },
    )
    output |= {
        "Momentum": {
            'model'   : momentum_model,
            'results' : momentum_results
        }
    }

    compress_pickle.dump(
        {
            "split_indices" : {
                "train"      : indices[0],
                "validation" : indices[1],
            } 
        } | output,
        results_path / "model_theta_results.mat",
    )

@click.command()
@click.option("--results-path", default="../results")
@click.option("--run-config")
@click.option("--rat", type=click.Choice(hseio.RAT_NAMES))
def main(results_path: str, run_config: str, rat: str):
    results_path = Path(results_path)
    run_config   = Path(run_config)
    with open(run_config, 'r') as f:
        parameters = json.loads(f.read())
    
    results_path = results_path / parameters.get("name")

    for session in os.listdir(results_path / rat):
        results_dir = results_path / rat / session
        track_type  = session[:-1]
        session_n   = int(session[-1])

        print(f"Training models on {rat} {session}")
        rat_data = compress_pickle.load(results_dir / "preprocessed_data.pkl.gz")

        if not parameters.get("ignore_theta", False):
            run_theta_models(
                rat_data['raw_data'],
                rat_data['placefield_data'],
                rat_data['theta_data'],
                parameters,
                results_dir
            )

        if not parameters.get("ignore_ripple", False):
            raise NotImplementedError

if __name__ == "__main__":
    main()