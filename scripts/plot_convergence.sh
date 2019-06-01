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

trace="$1"
./run.sh scripts/plot_convergence.py "$trace" \
         --trials "${trials[@]}" \
