import os 
import sys 
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 
import numpy as np
import torch

import hippocampalseq as hse
import hippocampalseq.utils as hseu
import hippocampalseq.preprocessing as hsep
import hippocampalseq.models as hsem
import hippocampalseq.plotting as hsepl

theta_time_window_s  = 60 / 1000
theta_time_window_advance_s  = 60 / 1000

bin_size = 2
data_path = os.path.realpath("../data")
rat_name = "Janni"
session = 2
track_type = "Open"
results_path = f"../results/{rat_name}/{track_type}{session}"

if track_type == "Linear":
    environment_size = None
else:
    environment_size =  hsep.EnvironmentSize((0,200), (0,200))

nplot = 20

(
    raw_data,
    place_field_data,
) = hse.load_raw_data(
    data_path,
    rat_name,
    session,
    track_type=track_type,
    environment_size=environment_size,
    bin_size_cm=bin_size,
    placefield_kwargs = {
        'place_field_posterior' : False,
        'velocity_cutoff'       : 5.0,
        "flatten_linear"        : True,
    },

)

theta_data = hse.process_theta(
    raw_data,
    place_field_data, 
    theta_kwargs = {
        'time_window_s'         : theta_time_window_s,
        'time_window_advance_s' : theta_time_window_advance_s,
        'velocity_cutoff'       : 10.0
    }
)

np.random.seed(42)

indices = hseu.train_test_valid_split(len(theta_data.spikes), 0.8, 0.1)

place_fields = place_field_data.place_fields[place_field_data.place_cell_ids]

true_velocity = [
    gt[[f'V_{d}' for d in raw_data.environment_size.axes()]].values
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

def run_model(
        model_name: str,
        model_kwargs: dict[str, Any],
        model_fit_kwargs: dict[str, Any] = {},
        model_pred_kwargs: dict[str, Any] = {}
    ):
    MODELS = {
        "MAP"                  : hsem.BayesianMAP,
        "Momentum"             : hsem.Momentum,
        "MomentumVelocity"     : hsem.MomentumVelocity,
        "MomentumVelocityBias" : hsem.MomentumVelocityBias,
        "CANNDynamics"         : hsem.CANNDynamics,
        "CANNDynamicsBias"     : hsem.CANNDynamicsSpikes
    }
    print(f"{model_name} running")
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

def plot_model(
        true_trajectories: list[np.ndarray], 
        model: hsem.StateSpace, 
        decoded: hsem.StateSpaceResults, 
        plot_num: int,
        loglike: list[float],
        validation_loglike: list[float],
        aic: float,
        bic: float,
        loglike_full: float
    ):
    trajectory = decoded.smoothed_mean[plot_num][:,model.latent_dim:]
    cum_prob = decoded.cumulative_probabilities[plot_num]

    figsize = (
        5*(4 if globals()['track_type'] == 'Open' else 1),
        5*(1 if globals()['track_type'] == 'Open' else 2)
    )
    plt.figure(figsize=figsize, dpi=300)
    plt.subplot(1,4,1)
    plt.title("True Trajectory")
    hsepl.plot_trajectories({
            'True Trajectory': true_trajectories[plot_num],
            'Momentum model': trajectory
        },
        environment_size=environment_size
    )
    if globals()['track_type'] == 'Open':
        plt.gca().set_aspect('equal')
    plt.subplot(1,4,2)
    plt.title("Cumulative Probability")
    if cum_prob.shape[-1] > 1:
        cum_prob = cum_prob.T
    plt.imshow(cum_prob, origin='lower', cmap='hot')
    plt.subplot(1,4,3)
    plt.title("Log-likelihood")
    plt.plot(loglike)
    plt.subplot(1,4,4)
    plt.title("Validation Log-likelihood")
    plt.plot(validation_loglike)

    print(f"BIC: {bic}")
    print(f"AIC: {aic}")
    print(f"Log-likelihood: {loglike_full}")
    print(f"L2 Error: {hsem.trajectory_error_posterior(hseu.atleast_3d(true_trajectories[plot_num]),trajectory.numpy(),)[1]}")
    print(f"Model decay: {model.decay} -> {torch.exp(model.decay)}")
    print(f"Model diffusion: {model.diffusion} -> {torch.exp(model.diffusion)}")

    plt.show()
    plt.savefig(f"{model.name()}_{plot_num}.png", dpi=300)
    plt.close()

import compress_pickle

names = []
aic = []
bic = []

#momentum_model,momentum_results = run_model(
#    "Momentum",
#    {
#        'dt'               : theta_time_window_s,
#        'bin_size'         : bin_size,
#        'environment_size' : raw_data.environment_size,
#        'place_fields'     : place_fields,
#        'spikemat_train'   : sp_train,
#        'spikemat_valid'   : sp_valid,
#    }, 
#    {},
#    {
#        'spikemats' : sp_test,
#    }
#)
#plot_model(
#    tp_train, 
#    momentum_model, 
#    momentum_results['training'], 
#    nplot,
#    momentum_results['training'].loglike,
#    momentum_results['validation'].loglike,
#    momentum_results['training'].aic,
#    momentum_results['training'].bic,
#    momentum_results['training'].loglike_full
#)
#compress_pickle.dump(
#    {
#        "model" : momentum_model,
#        "results" : momentum_results
#    },
#    "momentum.pkl.gz"
#)
with open('momentum.pkl.gz', 'rb') as f:
    pkl = compress_pickle.load(f)
    momentum_model = pkl['model']
    momentum_results = pkl['results']

aic.append(momentum_results['training'].aic)
bic.append(momentum_results['training'].bic)
names.append(momentum_model.name())
del momentum_model
del momentum_results
print("Model saved\n")

# momentumv_model,momentumv_results = run_model(
#     "MomentumVelocity",
#     {
#         'dt'               : theta_time_window_s,
#         'bin_size'         : bin_size,
#         'environment_size' : raw_data.environment_size,
#         'place_fields'     : place_fields,
#         'spikemat_train'   : sp_train,
#         'spikemat_valid'   : sp_valid,
#         'velocity_type'    : 'true'
#     }, 
#     {
#         'Xtrain' : tv_train,
#         'Xvalid' : tv_valid,
#     },
#     {
#         'spikemats' : sp_test,
#         'Xtest'     : tv_test,
#     }
# )

# plot_model(
#     tp_train, 
#     momentumv_model, 
#     momentumv_results['training'], 
#     nplot,
#     momentumv_results['training'].loglike,
#     momentumv_results['validation'].loglike,
#     momentumv_results['training'].aic,
#     momentumv_results['training'].bic,
#     momentumv_results['training'].loglike_full
# )
# compress_pickle.dump(
#     {
#         "model" : momentumv_model,
#         "results" : momentumv_results
#     },
#     "momentumv.pkl.gz"
# )
with open('momentumv.pkl.gz', 'rb') as f:
    pkl = compress_pickle.load(f)
    momentumv_model = pkl['model']
    momentumv_results = pkl['results']
aic.append(momentumv_results['training'].aic)
bic.append(momentumv_results['training'].bic)
names.append(momentumv_model.name())
del momentumv_model
del momentumv_results
print("Model saved\n")

# momentumvb_model,momentumvb_results = run_model(
#     "MomentumVelocityBias",
#     {
#         'velocity_train'   : tv_train,
#         'velocity_valid'   : tv_valid,
#         'bias_fn'          : 'linear',
#         'dt'               : theta_time_window_s,
#         'bin_size'         : bin_size,
#         'environment_size' : raw_data.environment_size,
#         'place_fields'     : place_fields,
#         'spikemat_train'   : sp_train,
#         'spikemat_valid'   : sp_valid,
#     }, 
#     {
#     },
#     {
#         'spikemats' : sp_test,
#         'Xtest'     : tv_test
#     }
# )

# plot_model(
#     tp_train, 
#     momentumvb_model, 
#     momentumvb_results['training'], 
#     nplot,
#     momentumvb_results['training'].loglike,
#     momentumvb_results['validation'].loglike,
#     momentumvb_results['training'].aic,
#     momentumvb_results['training'].bic,
#     momentumvb_results['training'].loglike_full
# )
# compress_pickle.dump(
#     {
#         "model" : momentumvb_model,
#         "results" : momentumvb_results
#     },
#     "momentumvb.pkl.gz"
# )
with open('momentumvb.pkl.gz', 'rb') as f:
    pkl = compress_pickle.load(f)
    momentumvb_model = pkl['model']
    momentumvb_results = pkl['results']

aic.append(momentumvb_results['training'].aic)
bic.append(momentumvb_results['training'].bic)
names.append(momentumvb_model.name())
del momentumvb_model
del momentumvb_results
print("Model saved\n")

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
        'Xtest'     : tv_test
    }
)

plot_model(
    tp_train, 
    cann_model, 
    results['training'], 
    nplot,
    cann_results['training'].loglike,
    cann_results['validation'].loglike,
    cann_results['training'].aic,
    cann_results['training'].bic,
    cann_results['training'].loglike_full
)
compress_pickle.dump(
    {
        "model" : cann_model,
        "results" : cann_results
    },
    "cann.pkl.gz"
)
#with open('cann.pkl.gz', 'rb') as f:
#    pkl = compress_pickle.load(f)
#    cann_model = pkl['model']
#    cann_results = pkl['results']

aic.append(cann_results['training'].aic)
bic.append(cann_results['training'].bic)
names.append(cann_model.name())
print("Model saved")
del cann_model
del cann_results

plt.bar(names, aic)
plt.xlabel("Model")
plt.ylabel("AIC")
plt.title("AIC for different models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("aic.png")

plt.bar(names, bic)
plt.xlabel("Model")
plt.ylabel("BIC")
plt.title("BIC for different models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("bic.png")