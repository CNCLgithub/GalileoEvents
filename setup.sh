#!/bin/bash

# This script will setup the project environment

# Change any of these values as you see fit.
# For initial run, all should be set to true.
BUILDCONT=false
BUILDENV=true

. load_config.sh


SING="${ENV[path]}"
CONT="${ENV[cont]}"
ENVPATH="${ENV[env]}"

# 1) Create the singularity container (requires sudo)
if [ $BUILDCONT = true ]; then
    if [ -f "$CONT" ]; then
        echo "Older container found...removing"
        rm -f "$CONT"
    fi
    if [ ! -f "blender.tar.bz2" ]; then
        wget "https://www.dropbox.com/s/3f39ste5xh6rjkt/blender.tar.bz2?dl=0" \
             -O "blender.tar.bz2"
    fi

    if [ ! -f "julia.tar.gz" ]; then
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.4-linux-x86_64.tar.gz \
             -O "julia.tar.gz"
    fi
    echo "Building container..."
    if [ ! -d $PWD/.tmp ]; then
        mkdir $PWD/.tmp
    fi
    SINGULARITY_TMPDIR=$PWD/.tmp sudo -E $SING build $CONT  Singularity
else
    echo "Not building container at ${CONT}"
fi

# Initialize python env
if [ $BUILDENV = true ]; then
    echo "Install python dependencies..."
    # Enter the container and run the command
    SING="${ENV['path']} exec --nv"
    mounts=(${ENV[mounts]})
    BS=""
    for i in "${mounts[@]}";do
        if [[ $i ]]; then
            BS="${BS} -B $i:$i"
        fi
    done
    ${SING} ${BS} ${CONT} bash -c "yes | conda create -p ${ENVPATH} python=3.6"
    chmod +x run.sh
    ./run.sh "python3 -m pip install -r requirements.txt"
    ./run.sh "python3 -m pip install -e ."

else
    echo "Not creating conda env at ${ENVPATH}"
fi

# Initiliazes julia depot
if [ ! -d ${ENV[julia_depot]} ]; then
    mkdir ${ENV[julia_depot]}
    ./run.sh "julia -e \"using Pkg; Pkg.instantiate()\""
else
    echo "Julia depot already exists at ${ENV[julia_depot]}"
fi
