#!/usr/bin/env bash
set -euo pipefail


# first make scene data
python scripts/stimuli/create_billiard_dataset.py # -> /scenes/pilot/...

# then convert to hdf5
[ -f "/databases/billiard.hdf5" ] && rm  "/databases/billiard.hdf5"
h5data-create "/scenes/billiard" -o "/databases/" # -> /databases/3ball_change.hdf5