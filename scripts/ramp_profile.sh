#!/bin/bash

# Runs ramp_profiling on ratios

. load_config.sh

ratio_file="${PATHS[scenes]}/mass_ratios_3.csv"
while IFS= read -r line; do
    IFS=',' read -ra ratios <<< "$line"
    ./run.sh scripts/ramp_profile.py "${ratios[@]}" --n_ramp 1
    ./run.sh scripts/ramp_profile.py "${ratios[@]}" --n_ramp 2
done < "$ratio_file"

./run.sh scripts/plot_ramp_profile.py
