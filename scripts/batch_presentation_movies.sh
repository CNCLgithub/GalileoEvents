#!/bin/bash

. load_config.sh

# get trial mappings
pres_file="${paths['databases']}/presentation_trials.csv"
orig=($(awk -f "\"*,\"*" '{print $5}' $pres_file | tail -n +2))
trials=()
# add incongruent pairs
for i in ${orig[@]}; do
    trials+=($i)
    trials+=($(($i+1)))
done

trace="$1"
renders="$2"
./run.sh scripts/movie_from_timings.py "${trace} ${renders}" \
         --mask --panorama 60
