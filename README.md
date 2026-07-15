# hippocampalseq
State-space modeling of hippocampal theta and replay sequences.

Inline LaTeX equations can be viewed by installing [this extension](https://marketplace.visualstudio.com/items?itemName=howcasperwhat.comment-formula) in VSCode/VSCodium.

## Submodules
### hippocampalseq
Contains two functions:
- `load_and_preprocess` which loads rat data with the format specified in `hippocampalseq.io` and preprocesses it.
- `process_ratdata` which takes the outputs from the previous function and runs analysis on them.

### hippocampalseq.analysis
Tools for analyzing neural data. Primarily focuses on theta rhythm.

### hippocampalseq.io
Contains tools for loading data formatted similarly to Brad Pfeiffer's data.
The required format for the data is as follows:
```
|
|-data
   |-[Rat name]
      |-[Open/Linear][session number]
         |-FILE.ncs - LFP data file
         |-Epochs.mat - Timestamps for rate behavioral epochs e.g. Run_Times/REM_Times/etc.
         |-Position_Data.mat - Contains time, (x,y) position, head direction in each respective column.
         |-Ripple_Events.mat - Contains start and stop points for previously decoded SWR events.
         |-Spike_Data.mat - Contains the id of the cell that spiked and what time it spiked at in respective columns.
```
If your data isn't formatted like this, ignore this submodule. If you format your data correctly,
you can still use the rest of the package.

### hippocampalseq.models
All state-space models are contained here.
- statespace.py - Basic state-space model implementing a kalman filter
- momentum.py - Subclassed model adding momentum to the variables.

### hippocampalseq.preprocessing
Contains preprocessing functions for tasks such as decoding place fields, segmented theta sequences, and more.

### hippocampalseq.plotting
Contains code used to generate figures in our paper and other plotting utilities.

### hippocampalseq.simulation
Contains code used to generate simulated data used in our analysis.

### hippoocampalseq.utils
Various code reused in all of our modules.

## jobs
The [jobs](./jobs/) folder contains code used in running the pipeline and generating figures end to end.
SLURM scripts used on UTSW's biohpc system are in here as well.

## notebooks
The [notebooks](./notebooks/) folder contains example jupyter notebooks for running the library
on Brad Pfeiffer's data.

## Installation
```
conda env create -f environment.yml
conda activate hippocampalseq
```

## Todo:
- Plotting in analyze_theta.py
- Decoding error implementation