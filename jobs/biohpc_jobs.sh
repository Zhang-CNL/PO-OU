#!/bin/bash
#SBATCH --job-name=hippocampalseq_models
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=pipeline.log
#SBATCH --error=pipeline.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

module load python/3.13.0
source $(conda info --base)/etc/profile.d/conda.sh

conda run -n hippocampalswr python run_pipeline.py $@