#!/bin/bash

sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear 
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --approximation-method 'iterative'

sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5 --approximation-method 'iterative'
