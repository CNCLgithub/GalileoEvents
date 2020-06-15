#!/usr/bin/env bash
set -euo pipefail


# first make scene data
python scripts/stimuli/3ball_change.py # -> /scenes/pilot/...

# then convert to hdf5
[ -f "/databases/3ball_change.hdf5" ] && rm  "/databases/3ball_change.hdf5"
h5data-create "/scenes/3ball_change" -o "/databases/" # -> /databases/3ball_change.hdf5
