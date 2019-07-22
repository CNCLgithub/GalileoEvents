#!/bin/bash


. load_config.sh

scenes=$(find "${PATHS[scenes]}" -mindepth 2 -maxdepth 2 -type f)
singleramp=$(echo "$scenes" | grep -E -e ".+\/1_.+\/[0-9]+_[0-9]+\.json")
doubleramp=$(echo "$scenes" | grep -E -e ".+\/2_.+\/[0-9]+_[0-9]+\.json")
positions=(1 39)
out="stimuli"
# Change mode for single or double ramp
# mode=$singleramp
mode=$doubleramp
for pos in "${positions[@]}";do
    rexp=".+\/${pos}_[0-9]+\.json"
    srcs=($(echo "$mode" | grep -E -e "$rexp"))
    ./scripts/render_stimuli.py --mode "default" --resolution 854 480 --out "$out" --run "batch" \
                                "${srcs[@]}"
done

