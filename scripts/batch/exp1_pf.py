#!/usr/bin/env python

""" Performs offline inference for Exp1 PF """

import os
import argparse
from slurmpy import sbatch

from galileo_ramp.exp1_dataset import Exp1Dataset

def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', type = str,
                        default = '/databases/exp1.hdf5',
                        help = 'Path to scenes')
    parser.add_argument('--particles', type = int,
                        default = 1,
                        help = 'Number of particles')
    parser.add_argument('--obs_noise', type = float,
                        default = 0.1,
                        help = 'observation noise')
    parser.add_argument('--chains', type = int,
                        default = 1,
                        help = 'Number of chains')
    args = parser.parse_args()

    # create out dir early to prevent conflicts
    out_path = '/traces/exp1_p_{0:d}_n_{1:.2f}'.format(args.particles, 
                                                       args.obs_noise)
    os.path.isdir(out_path) or os.mkdir(out_path)


    dataset = Exp1Dataset(args.src)


    # we only need the test trials
    njobs = 120

    tasks = [(str(i),) for i in range(njobs)]
    kwargs = ['--particles {0:d}'.format(args.particles),
              '--obs_noise {0:f}'.format(args.obs_noise),
              '--chains {0:d}'.format(args.chains)]

    interpreter = '#!/bin/bash'
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '40',
        'partition' : 'short',
        'requeue' : None,
    }
    path = '/project/scripts/inference/exp1_pf.jl'
    sys_img = '/project/sys_galileo_ramp.so'
    func = 'bash {0!s}/run.sh julia -J sys.so --compiled-modules=no {1!s}'
    func = func.format(os.getcwd(), path)
    # func = 'bash {0!s}/run.sh julia --sysimage {1!s} {2!s}'
    # func = func.format(os.getcwd(), sys_img, path)
    batch = sbatch.Batch(interpreter, func, tasks, kwargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    batch.run(n = njobs, check_submission = True)

if __name__ == '__main__':
   main()
