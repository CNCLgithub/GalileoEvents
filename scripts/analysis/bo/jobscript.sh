#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=distributed_bo
#SBATCH --time=00-00:02:00
#SBATCH --output=distributed_bo.output  # output and error messages go to this file

srun ./run.sh julia scripts/analysis/bo/distributed_bo.jl
