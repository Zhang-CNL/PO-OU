#!/bin/bash
#SBATCH --job-name=hippocampalseq_models
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --time=20-00:00:00                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=./logs/pipeline.%j.log
#SBATCH --error=./logs/pipeline.%j.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

module load python/3.13.0
source $(conda info --base)/etc/profile.d/conda.sh

RAT_NAMES=("Harpy" "Imp" "Janni" "Naga")

DATETIME=$(date +"%Y-%m-%d")
RESULTS_PATH="../results/${DATETIME}-${SLURM_ARRAY_JOB_ID}/"
CHECKPOINT_PATH="../checkpoints/${DATETIME}-${SLURM_ARRAY_JOB_ID}/"

#conda run -n hippocampalswr python -u run_pipeline.py $@
conda activate hippocampalswr
python -u run_pipeline.py --rats "${RAT_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --results-path $RESULTS_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    $@
#python -u run_pipeline.py $@