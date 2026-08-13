#!/bin/bash

rm ../logs/*.log

for i in {1..6}; do
    sbatch ./analysis.sh --run-config ./profiles/theta_profile${i}.json
done