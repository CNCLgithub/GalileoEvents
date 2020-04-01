#!/bin/bash

. load_config.sh


build=${1:-"false"}
conda=${2:-"false"}
julia=${3:-"false"}


if [ "$build" = "true" ]; then
    echo "building..."
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity
fi

if [ "$conda" = "true" ]; then
    echo "Setting up the Conda environment"
    singularity exec ${ENV[cont]} bash -c "yes | conda create -p $PWD/${ENV[env]} python=3.6"
    ./run.sh python -m pip install -r requirements.txt
fi

if [ "$julia" = "true" ]; then
    echo "Setting up julia env..."
    ./run.sh julia -e '"using Pkg; Pkg.instantiate()"'
fi
