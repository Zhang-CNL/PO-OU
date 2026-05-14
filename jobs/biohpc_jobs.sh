#!/bin/bash
#SBATCH --job-name=serialJob 
#SBATCH --partition=256GBv2
#SBATCH --nodes=1
#SBATCH --time=0-00:00:30                                 # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=pipeline.log
#SBATCH --error=pipeline.error.log
#SBATCH --mail-user=armand.rathgeb@utsouthwestern.edu
#SBATCH --mail-type=ALL

module load python/3.13.0
conda activate hippocampalswr

python run_pipeline.py $@