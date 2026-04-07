import os
import numpy as np

import hippocampalseq.preprocessing as hsep
import hippocampalseq.models as hsem
import hippocampalseq.utils as hseu


def run_pipeline(data_path: str, rat_name: str, session: int, dt: float, model: str, seed: int|None = 42, results_dir: str = "./results/"):
    assert model in ['momentum', 'momentum_gridsearch', 'bayes_map']

    results_dir = os.path.join(results_dir, f"{rat_name}-{session}", model)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    rat_data = hsep.load_and_preprocess(
        os.path.realpath(data_path), 
        rat_name, 
        session, 
        time_window_ms=dt
    )
    if model == 'momentum':
        model = hsem.Momentum(rat_data)
        results, train_ll, test_ll = model.em(n_iter=100000, emtol=1e-3, seed=seed)
        hseu.save_pickle(model, results_dir / "model.pkl")
    elif model == 'momentum_gridsearch':
        model = hsem.MomentumGridSearch(rat_data)
    
        hseu.save_pickle(model, results_dir / "model.pkl")
    elif model == 'bayes_map':
        trajectories = hsem.bayesian_decoding(
            rat_data.place_field_data.place_fields[rat_data.place_field_data.place_cell_ids],
            rat_data.ripple_data.spikemats_ripple, 
            dt
        )
