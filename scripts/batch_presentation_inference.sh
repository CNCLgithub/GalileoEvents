#!/bin/bash

. load_config.sh

# get trial mappings
pres_file="${PATHS['databases']}/presentation_trials.csv"
orig=($(awk -F "\"*,\"*" '{print $5}' $pres_file | tail -n +2))
trials=()
# add incongruent pairs
for i in ${orig[@]}; do
    trials+=($i)
    trials+=($(($i+1)))
done

echo $trials
dataset="combinded_05_21_19_143824.hdf5"
params="combinded_05_21_19_143824_pf_results_190529_142809"
# trial="$1"
./run.sh scripts/particle_filter.py --dataset "$dataset" --chains 10 \
         --particles 20 --steps 10 \
         --trials "${trials[@]}" \
         --parameters "${params}/parameters.json" \
         --slurm
         # --resample 0 --perturb 0 --mu 9.0 --sigma 0.001 \
