#!/bin/bash

rm ../logs/*.log

#sbatch ./models.sh --run-config ./profiles/ripple_profile.json
for i in {1..6}; do
    sbatch ./models.sh --run-config ./profiles/theta_profile${i}.json
done
