#!/bin/bash

# Process all replay, only one time-step analyzed
sbatch ./biohpc_jobs.sh --data-type replay --delta-t-ms 3 --time-step-ms 3 --place-field-posterior --velocity-cutoff 5

# Process theta with params similar to the Pfeiffer paper
sbatch ./biohpc_jobs.sh --data-type theta --delta-t-ms 10 --time-step-ms 5 --place-field-posterior --velocity-cutoff 10

# Process theta with different time steps corresponding to K&D and Pfeiffer
declare -a time_steps=(9 60 250)

for ${time_step} in "${time_steps[@]}"; do
    sbatch ./biohpc_jobs.sh --data-type theta --delta-t-ms ${time_step} --time-step-ms ${time_step} --place-field-posterior --velocity-cutoff 5
done