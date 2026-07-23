#!/bin/bash

#sbatch ./models.sh --run-config ./profiles/ripple_profile.json
sbatch ./models.sh --run-config ./profiles/theta_profile1.json
sbatch ./models.sh --run-config ./profiles/theta_profile2.json
sbatch ./models.sh --run-config ./profiles/theta_profile3.json
