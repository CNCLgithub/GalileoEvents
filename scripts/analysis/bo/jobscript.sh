#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=distributed_bo
#SBATCH --time=00-00:02:00  # Wall Clock time (dd-hh:mm:ss) [max of 14 days]
#SBATCH --output=distributed_bo.output  # output and error messages go to this file

srun --ntasks=1 ./run.sh julia scripts/analysis/bo/distributed_bo.jl
