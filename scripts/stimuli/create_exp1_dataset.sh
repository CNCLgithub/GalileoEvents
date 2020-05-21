#!/usr/bin/env bash
set -euo pipefail


# first make scene data
python scripts/stimuli/create_exp1_dataset.py # -> /scenes/pilot/...

# then convert to hdf5
[ -f "/databases/exp1.hdf5" ] && rm  "/databases/exp1.hdf5"
h5data-create "/scenes/exp1" -o "/databases/" # -> /databases/exp1.hdf5
