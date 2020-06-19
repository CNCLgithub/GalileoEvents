#!/usr/bin/env bash
set -euo pipefail

python scripts/stimuli/render_billiard.py "/datasets/billiards.hdf5" --run batch --mode default
