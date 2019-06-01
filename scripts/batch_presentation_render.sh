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

batch="${#trials[@]}"
dataset="combinded_05_21_19_143824.hdf5"
./run.sh python3 scripts/render_tower_pair.py --src "$dataset" \
         --run batch --batch "$batch" \
         --mode "default" \
         --trial "${trials[@]}"
