#!/usr/bin/env bash
set -euo pipefail


# first make scene data
python scripts/stimuli/create_3ball_scene.py # -> /scenes/pilot/...

# then convert to hdf5
[ -f "/databases/3ball.hdf5" ] && rm  "/databases/3ball.hdf5"
h5data-create "/scenes/3ball" -o "/databases/" # -> /databases/3ball.hdf5
