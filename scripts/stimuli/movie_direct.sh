#!/bin/bash

# Renders 120 frames each scene in Exp1

WRKDIR=${1:-"/renders/exp1"}

find $WRKDIR -mindepth 1 -maxdepth 1 -type d -exec \
    python scripts/stimuli/movie_direct.py '{}' \;

