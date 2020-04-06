#!/bin/bash

. load_config.sh


cont=${1:-""}
conda=${2:-""}
julia=${3:-""}

# container setup
[ -z "$cont" ] || [ "$cont" = "false" ] && echo "Not touching container"
[ "$cont" = "pull" ] && wget "https://yale.box.com/shared/static/i5vxp5xghtfb2931fhd4b0ih4ya62o2s.sif" -O "${ENV[cont]}" 
[ "$cont" = "build" ] || [ "$cont" = "true" ] && echo "building container" && \
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity


# conda setup
[ -z "$conda" ] || [ "$conda" = "false" ] && echo "Not touching conda"
[ "$conda" = "build" ] || [ "$conda" = "true" ] && echo "building conda env" && \
    singularity exec ${ENV[cont]} bash -c "yes | conda create -p $PWD/${ENV[env]} python=3.6" && \
    ./run.sh python -m pip install -r requirements.txt

# julia setup

[ -z "$julia" ] || [ "$julia" = "false" ] && echo "Not touching julia"
[ "$julia" = "build" ] || [ "$julia" = "true" ] && echo "building julia env" && \
    ./run.sh julia -e '"using Pkg; Pkg.instantiate()"'
