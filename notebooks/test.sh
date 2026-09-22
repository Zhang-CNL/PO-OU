#!/bin/bash
#SBATCH --job-name=hippocampalseq_analysis
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --time=20-00:00:00                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=../logs/pipeline.%j.log
#SBATCH --error=../logs/pipeline.%j.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

module load python/3.14.0
source $(conda info --base)/etc/profile.d/conda.sh

conda activate hippocampalswr
python -u run_model.py