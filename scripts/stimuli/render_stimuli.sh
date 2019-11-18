#!/bin/bash


. load_config.sh

scenes=$(find "${PATHS[scenes]}" -mindepth 2 -maxdepth 2 -type f)
singleramp=$(echo "$scenes" | grep -E -e ".+\/1_.+\/[0-9]+_[0-9]+\.json")
doubleramp=$(echo "$scenes" | grep -E -e ".+\/2_.+\/[0-9]+_[0-9]+\.json")
singlepositions=(1 39)
doublepositions=(17 30)
out="stimuli"

# single ramp
for pos in "${singlepositions[@]}";do
    rexp=".+\/${pos}_[0-9]+\.json"
    srcs=($(echo "$singleramp" | grep -E -e "$rexp"))
    ./scripts/render_stimuli.py --mode "default" --resolution 854 480 --out "$out" --run "batch" \
                                "${srcs[@]}"
done

# double ramp
for pos in "${doublepositions[@]}";do
    rexp=".+\/${pos}_[0-9]+\.json"
    srcs=($(echo "$doubleramp" | grep -E -e "$rexp"))
    ./scripts/render_stimuli.py --mode "default" --resolution 854 480 --out "$out" --run "batch" \
                                "${srcs[@]}"
done

