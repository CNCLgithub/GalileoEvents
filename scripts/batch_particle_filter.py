#!/usr/bin/env python3
"""
Batches inference calls across a series of given trials.
"""
import os
import argparse
import datetime
from pprint import pprint
from itertools import repeat

import numpy as np
from slurmpy import sbatch
from galileo_ramp.utils import config

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']

script = os.path.join(root, 'scripts', 'particle_filter.py')

def submit_sbatch(trials, params, out, size = 1000):

    njobs = min(size, len(trials))

    interpreter = '#!/bin/bash'
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './run.sh python3 -W ignore {0!s}'.format(script)
    tasks = list(zip(trials, repeat(params)))
    kargs=['--out {0!s}'.format(out)]
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '1-0',
        'qos' : 'use-everything',
        # 'qos' : 'tenenbaum',
        'requeue' : None,
        # 'output' : '/dev/null'
        'output' : os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out')
    }
    batch = sbatch.Batch(interpreter, func, tasks, kargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    batch.run(n = njobs, check_submission = False)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trials', type = str, nargs = '+',
                        help = 'path to scene files')
    parser.add_argument('parameters', type = str,
                        help = 'path to parameter json')
    parser.add_argument('--resume', type = str,
                        help = 'directory to resume traces')

    args = parser.parse_args()

    if args.resume is None:
        name = datetime.datetime.now().strftime("%m-%d-%y_%H%M%S")
        name = args.parameters[:-5] + '_' + name
        out = os.path.join(CONFIG['PATHS', 'traces'], name)
        if not os.path.isdir(out):
            os.mkdir(out)
    else:
        out = args.resume


    submit_sbatch(args.trials, args.parameters, out)

if __name__ == '__main__':
    main()
