#!/bin/bash

. load_config.sh

# Define the path to the container and conda env
CONT="${ENV['cont']}"

# Parse the incoming command
COMMAND="$@"

# Enter the container and run the command
SING="${ENV['path']} exec --nv"
mounts=(${ENV[mounts]})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
        BS="${BS} -B $i:$i"
    fi
done

${SING} ${BS} ${CONT} bash -c "source activate ${PWD}/${ENV[env]} \
        && export JULIA_DEPOT_PATH=${ENV[julia_depot]} \
        && export JULIA_PROJECT=${PWD} \
        && cd ${PWD} \
        && exec $COMMAND \
        && source deactivate"
