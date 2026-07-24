#!/bin/bash
#SBATCH --job-name=hippocampalseq_models
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --time=20-00:00:00                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=../logs/pipeline.%j.log
#SBATCH --error=../logs/pipeline.%j.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

module load python/3.13.0
source $(conda info --base)/etc/profile.d/conda.sh

RAT_NAMES=("Ettin" "Harpy" "Imp" "Janni" "Naga")

DATETIME=$(date +"%Y-%m-%d")
RESULTS_PATH="/project/bioinformatics/WZhang_lab/shared/theta_momentum/${DATETIME}"

conda activate hippocampalswr
# Runs each model and saves the processed data to {RESULTS_PATH}/{RAT_NAME}/{SESSION}
# Model results are saved to {RESULTS_PATH}/{RAT_NAME}/{SESSION}/{MODEL}
python -u models.py --rats "${RAT_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --results-path $RESULTS_PATH $@
