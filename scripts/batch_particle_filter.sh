#!/bin/bash

. load_config.sh

scenes=$(find "${PATHS[scenes]}" -mindepth 2 -maxdepth 2 -type f)
singleramp=$(echo "$scenes" | grep -E -e ".+\/1_.+\/[0-9]+_[0-9]+\.json")
positions='([1]|39)'

rexp=".+\/${positions}_[0-9]+\.json"
trials=($(echo "$singleramp" | grep -E -e "$rexp"))


# ./scripts/batch_particle_filter.py "${trials[@]}" \
#                                    "factor-smc-params.json"

./scripts/batch_particle_filter.py "${trials[@]}" \
                                   "gibbs-smc-params.json"
