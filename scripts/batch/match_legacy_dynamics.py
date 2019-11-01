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

script = os.path.join(root, 'scripts', 'validation',
                      'match_legacy_dynamics.py')

def submit_sbatch(trials, size = 1000):

    njobs = min(size, len(trials))

    interpreter = '#!/bin/bash'
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './run.sh python3 -W ignore {0!s}'.format(script)
    tasks = [(t,) for t in trials]
    kargs=[]
    extras = []
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '2GB',
        'time' : '30',
        'partition' : 'scavange',
        'requeue' : None,
        'output' : os.path.join(CONFIG['PATHS', 'sout'], 'slurm-%A_%a.out')
    }
    batch = sbatch.Batch(interpreter, func, tasks, kargs, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=njobs)))
    # batch.run(n = njobs, check_submission = False)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trials', type = str,
                        help = 'path to scene files')

    args = parser.parse_args()

    files = glob(os.path.join(args.trials, '*.json'))


    submit_sbatch(files)

if __name__ == '__main__':
    main()
