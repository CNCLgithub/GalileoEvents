#!/bin/bash

# This script will setup the project environment

# Change any of these values as you see fit.
# For initial run, all should be set to true.
# "pull" : Download from host
# "build" : Build locally
BUILDCONT="pull"
BUILDENV=true

. load_config.sh


SING="${ENV[path]}"
CONT="${ENV[cont]}"
ENVPATH="${ENV[env]}"

DEPPATH="https://www.dropbox.com/sh/exloazrievnjvey/AADXtso1A4WaPKQ09LX1alFAa?dl=0"
BLENDPATH="https://www.dropbox.com/s/rg5hhphs4hdxzun/blender.tar.bz2?dl=0"
JULIAPATH="https://www.dropbox.com/s/w04yhfn3jp9sndd/julia.tar.gz?dl=0"

# 1) Create the singularity container (requires sudo)
if [ $BUILDCONT = "pull" ]; then
    wget "$DEPPATH" -O "_env.zip"
    unzip "_env.zip"
    echo "Moving container..."
    mv "cont" "$SING"
elif [ $BUILDCONT = "build" ]; then
    if [ ! -f "blender.tar.bz2" ]; then
        wget "$BLENDPATH" -O "blender.tar.bz2"
    fi
    if [ ! -f "julia.tar.gz" ]; then
        wget "$JULIAPATH" -O "julia.tar.gz"
    fi
    echo "Building container...(REQUIRES ROOT)"
    if [ ! -d $PWD/.tmp ]; then
        mkdir $PWD/.tmp
    fi
    SINGULARITY_TMPDIR=$PWD/.tmp sudo -E $SING build $CONT  Singularity
else
    echo "Not touching container at ${CONT}"
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
