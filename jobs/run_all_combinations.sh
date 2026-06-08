#!/bin/bash

sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear 
sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --place-field-posterior
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --place-field-posterior
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --normalize

# Recreate some bits from the K&D paper
sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --place-field-posterior --theta-delta-t-ms 9 --theta-time-step-ms 9 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --place-field-posterior --theta-delta-t-ms 9 --theta-time-step-ms 9 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --place-field-posterior --theta-delta-t-ms 60 --theta-time-step-ms 60 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --place-field-posterior --theta-delta-t-ms 60 --theta-time-step-ms 60 --velocity-cutoff 5

sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model map --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5 --place-field-posterior
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5
sbatch ./biohpc_jobs.sh --model momentum --rerun --skip-linear --theta-delta-t-ms 250 --theta-time-step-ms 250 --velocity-cutoff 5 --normalize
