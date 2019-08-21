#!/bin/bash

# Runs ramp_profiling on ratios

. load_config.sh

ratio_file="${PATHS[scenes]}/mass_ratios_3.csv"
while IFS= read -r line; do
    IFS=',' read -ra ratios <<< "$line"
    scripts/ramp_profile.py "${ratios[@]}" --n_ramp 1
    scripts/ramp_profile.py "${ratios[@]}" --n_ramp 2
done < "$ratio_file"

scripts/plot_ramp_profile.py
