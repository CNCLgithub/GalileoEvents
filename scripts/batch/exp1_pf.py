#!/usr/bin/env python

""" Performs offline inference for Exp1 PF """

import os
import argparse
from slurmpy import sbatch


def main():

    parser = argparse.ArgumentParser(
        description = 'Evaluates a batch of blocks for stimuli generation.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('src', type = str,
                        help = 'Path to scenes')
    parser.add_argument('--particles', type = int,
                        default = 1,
                        help = 'Number of particles')
    parser.add_argument('--obs_noise', type = float,
                        default = 0.1,
                        help = 'Number of particles')
    args = parser.parse_args()

    dataset = Exp1Dataset(args.src)


    njobs = len(dataset)

    tasks = [(str(i)) for i in range(len(dataset))]
    kwargs = ['--particles {0:d}'.format(args.particles),
              '--obs_noise {0:f}'.format(args.obs_noise)]

    interpreter = '#!/bin/bash'
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '1GB',
        'time' : '30',
        'partition' : 'scavenge',
        'requeue' : None,
    }
    path = '/project/scripts/inference/exp1_pf.jl'
    sys_img = '/project/sys_galileo_ramp.so'
    func = 'bash {0!s}/run.sh julia --sysimage {1!s} {2!s}'
    func = func.format(os.getcwd(), sys_img, path)
    batch = sbatch.Batch(interpreter, func, tasks, kwargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    batch.run(n = njobs, check_submission = True)

if __name__ == '__main__':
   main()
