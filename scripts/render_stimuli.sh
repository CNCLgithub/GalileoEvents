#!/bin/bash


. load_config.sh

scenes=$(find "${PATHS[scenes]}" -mindepth 2 -maxdepth 2 -type f)
singleramp=$(echo "$scenes" | grep -E -e ".+\/1_.+\/[0-9]+_[0-9]+\.json")
positions=(1 39)
out="stimuli"
for pos in "${positions[@]}";do
    rexp=".+\/${pos}_[0-9]+\.json"
    srcs=($(echo "$singleramp" | grep -E -e "$rexp"))
    ./scripts/render_stimuli.py --mode "motion" --out "$out" --run "batch" \
                                "${srcs[@]}"
done

