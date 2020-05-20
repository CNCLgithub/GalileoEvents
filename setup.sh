#!/bin/bash

. load_config.sh

usage="$(basename "$0") [targets...] -- setup an environmental component of the project according to [default|local].conf

supported targets:

    cont_[pull|build] : either pull the singularity container or build from scratch
    conda : build the conda environment
    julia: build julia environment
"

[ $# -eq 0 ] || [[ "${@}" =~ "help" ]] && echo "$usage"

# container setup
[[ "${@}" =~ "cont_build" ]] || [[ "${@}" =~ "cont_pull" ]] || echo "Not touching container"
[[ "${@}" =~ "cont_pull" ]] && echo "pulling container" && \
    wget "https://yale.box.com/shared/static/i5vxp5xghtfb2931fhd4b0ih4ya62o2s.sif" \
    -O "${ENV[cont]}"
[[ "${@}" =~ "cont_build" ]] && echo "building container" && \
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity


# conda setup
[[ "${@}" =~ "conda" ]] || echo "Not touching conda"
[[ "${@}" =~ "conda" ]] && echo "building conda env" && \
    singularity exec ${ENV[cont]} bash -c "yes | conda create -p $PWD/${ENV[env]} python=3.6" && \
    ./run.sh python -m pip install -r requirements.txt && \

# julia setup
[[ "${@}" =~ "julia" ]] || echo "Not touching julia"
[[ "${@}" =~ "julia" ]] && echo "building julia env" && \
    ./run.sh julia -e '"using Pkg; Pkg.instantiate()"'
