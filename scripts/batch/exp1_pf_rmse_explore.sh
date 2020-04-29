#!/usr/bin/env bash

# Runs inference for
#scripts/batch/exp1_pf.py --particles 4 --obs_noise 0.01 --chains 20
#scripts/batch/exp1_pf.py --particles 4 --obs_noise 0.02 --chains 20
#scripts/batch/exp1_pf.py --particles 4 --obs_noise 0.05 --chains 20
# scripts/batch/exp1_pf.py --particles 100 --obs_noise 0.05 --chains 20
scripts/batch/exp1_pf.py --particles 100 --obs_noise 0.075 --chains 20
scripts/batch/exp1_pf.py --particles 100 --obs_noise 0.10 --chains 20
#scripts/batch/exp1_pf.py --particles 300 --obs_noise 0.01
#scripts/batch/exp1_pf.py --particles 300 --obs_noise 0.02
#scripts/batch/exp1_pf.py --particles 300 --obs_noise 0.05
