# hippocampalseq
State-space modeling of hippocampal theta and replay sequences.

## Submodules
### hippocampalseq.models
All state-space models are contained here.
- statespace.py - Basic state-space model implementing a kalman filter
- momentum.py - Subclassed model adding momentum to the variables.

### hippocampalseq.preprocessing
Preprocessing code ripped from https://github.com/DrugowitschLab/hippocampalseqDynamics/ and then modified.
Loading data can be done using the `load_and_preprocess()` function:
```py
import hippocampalseq.preprocessing as hsep
rat_data = hsep.load_and_preprocess("path/to/data", "RatName", session)
```

### hippoocampalseq.utils
Code for approximating place-fields as gaussians and more.

## notebooks
The [notebooks](./notebooks/) folder contains jupyter notebooks for running the library
on Brad Pfeiffer's data.

## Installation
```
conda env create -f environment.yml
conda activate hippocampalseq
```