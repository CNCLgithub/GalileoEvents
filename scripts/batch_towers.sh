#!/bin/bash

# really stable
# ./run.sh python3 scripts/noisy_tower_search.py --total 10 --base 2 2 \
#                  --force 1000.0 --noise 0.175 --batch 300 --slurm  H stable \
#                  --upper 0.2 --lower 0.5 --noisy 0.5

./run.sh python3 scripts/noisy_tower_search.py --total 10 --base 2 2 \
                 --force 1000.0 --noise 0.175 --batch 300 --slurm  L stable \
                 --upper 0.2 --lower 0.5 --noisy 0.5

# ./run.sh python3 scripts/noisy_tower_search.py --total 10 --base 2 2 \
#                  --force 1000.0 --noise 0.175 --batch 300 --slurm  H unstable \
#                  --upper 0.2 --lower 0.5 --noisy 0.5

# ./run.sh python3 scripts/noisy_tower_search.py --total 30 --base 2 2 \
#                  --out unstable_l_0.2_0.5_0.5_190516_021230 \
#                  --force 1000.0 --noise 0.175 --batch 200 --slurm  L unstable \
#                  --upper 0.2 --lower 0.5 --noisy 0.5
# somewhat unstable
# ./run.sh python3 scripts/noisy_tower_search.py --total 5 --base 2 2 \
#                  --force 250.0 --noise 0.2 --batch 100 --slurm  h stable \
#                  --upper 0.2 --lower 0.25 --noisy 0.3
# # unstable
# ./run.sh python3 scripts/noisy_tower_search.py --total 5 --base 2 2 \
#                  --force 250.0 --noise 0.2 --batch 100 --slurm  h stable \
#                  --upper 0.35 --lower 0.35 --noisy 0.4
