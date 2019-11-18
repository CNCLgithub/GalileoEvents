# Inference in Galileo 

## Outline

The inference pipeline is split into two sections:

1. The procedure definition under `inference/*.jl`
2. The parameterization and admistration of the procedure in `scripts/`


In addition, there are some auxillary modules under `galileo-ramp/inference` that allow 
a python `->` julia interface

## Inference Procedures

Algorithmic descriptions of the inference procedures can be found in `inference/algorithims.rst`


## Parameterization

The free parameters for each procedure are described in json files
(ie `factor-smc-params.json`)
that can be found under `CONFIG[traces]`.

These parameter files are used by `scripts/particle_filter.py` to run inference over a given 
trial.

To schedule batches of infernece across a trial dataset, `scripts/batch_particle_filter.py`
will dispacth a `SLURM` job script given a parameter file and a dataset. This can be interfaced at a high level using a companion batch script at `scripts/batch_particle_filter.sh`.

