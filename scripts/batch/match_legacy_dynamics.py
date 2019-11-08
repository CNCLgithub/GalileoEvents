#!/usr/bin/env python3
"""
Batches inference calls across a series of given trials.
"""
import os
import argparse
import datetime
from glob import glob
from pprint import pprint
from itertools import repeat

import numpy as np
from slurmpy import sbatch
from galileo_ramp.utils import config

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']


def submit_sbatch(trials, script, size = 1000):

    njobs = min(size, len(trials))

    interpreter = '#!/bin/bash'
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './run.sh python3 -W ignore {0!s}'.format(script)
    tasks = [(t,) for t in trials]
    kargs=[]
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '1GB',
        'time' : '360',
        'partition' : 'scavenge',
        'requeue' : None,
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
    parser.add_argument('--trials', type = str, default = 'legacy_converted',
                        help = 'path to scene files')
    parser.add_argument('--inference', type = str, default = 'mh',
                        help = 'inference procedure to apply')

    args = parser.parse_args()

    src_path = os.path.join(CONFIG['PATHS', 'scenes'], args.trials)
    files = glob(os.path.join(src_path, '*.json'))

    if args.inference == 'mh':
        script = os.path.join(root, 'scripts', 'validation',
                              'match_legacy_dynamics_mh.py')
    else:
        script = os.path.join(root, 'scripts', 'validation',
                              'match_legacy_dynamics.py')
    submit_sbatch(files, script)

if __name__ == '__main__':
    main()
