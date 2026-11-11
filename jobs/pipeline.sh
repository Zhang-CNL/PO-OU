#!/bin/bash
#SBATCH --job-name=hippocampalseq_analysis
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --array=1-4
#SBATCH --ntasks=1
#SBATCH --time=20-00:00:00                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=../logs/pipeline.%j.log
#SBATCH --error=../logs/pipeline.%j.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

set -e

#module load python/3.14.0
#source /etc/profile.d/conda.sh

RAT_NAMES=("Ettin" "Harpy" "Imp" "Janni") #"Naga")
RAT_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
RAT_NAME="${RAT_NAMES[$RAT_INDEX]}"

RESULTS_PATH="/project/bioinformatics/WZhang_lab/shared/theta_momentum/"

#conda activate hippocampalswr
# Runs each model and saves the processed data to {RESULTS_PATH}/{RAT_NAME}/{SESSION}
python -u preprocess.py --rat "$RAT_NAME" \
    --results-path "$RESULTS_PATH" "$@" 

python -u models.py --rat "$RAT_NAME" \
    --results-path "$RESULTS_PATH" "$@"

python -u individual_rat_analysis.py --rat "$RAT_NAME" \
    --results-path "$RESULTS_PATH" "$@"