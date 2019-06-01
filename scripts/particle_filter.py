#!/usr/bin/env python3
"""
Simple test script that runs inference in Gen through python
"""
import os
import glob
import h5py
import json
import argparse
import datetime
import numpy as np
from pprint import pprint
from slurmpy import sbatch
from dask import distributed
from dask_jobqueue import SLURMCluster

from mc.utils import  hdf5
from experiment.inference.execute import initialize
from experiment.dataset.particle_dataset import ParticleDataset

from utils import config
CONFIG = config.Config()

root = CONFIG['PATHS', 'root']
module_path = os.path.join(root, 'inference', 'smc.jl')

def get_gt(tower, blocks):
    substances = tower.extract_feature('substance')
    d = np.empty((len(blocks),), dtype = float)
    for idx,b in enumerate(blocks):
        d[idx] = substances[b-1]['density']
    return d

class PFEncoder(json.JSONEncoder):

    """ Encodes PF results for json """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(PFEncoder, self).default(obj)



def run_search(args):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    dataset = ParticleDataset(args['dataset'])

    # Retrieve trial info
    (obs, blocks), tower = dataset[args['trial']]

    smc = initialize(args['module'])
    print('RUNNING SMC')
    gt = get_gt(tower, blocks)
    mass_prior = np.array((args['mu'], args['sigma'], *args['bounds']))
    scene_args = (tower.serialize(), blocks, mass_prior, gt[0], ['unknown_mass'])
    inf_args = (args['particles'], args['steps'], args['resample'])
    results = smc(scene_args, inf_args, args['perturb'])
    results['gt'] = gt
    # dict containing: gt, xs, scores, estimates
    with open(args['out'], 'w') as f:
        json.dump(results, f, cls = PFEncoder)
    return args['out']

###############################################################################
# Helpers
###############################################################################

def chain_name(out, idx, c):
    s = os.path.join(out, 'trial_{0:d}_chain_{1:d}.json')
    return s.format(idx, c)

def load_default_parameters(args):
    """ Formats inputs from `argparse.ArgumentParser` for `smc.SMC`.

    Arguments:
        args: A `Namespace` collection

    Returns:
        A dictionary of relevant variables for inference.
    """
    t = vars(args)
    keys = ['mu', 'sigma', 'bounds', 'width', 'particles', 'resample',
            'steps', 'pos_proposal', 'pos_prior', 'module', 'perturb']
    return {k:t[k] for k in keys}

def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo block-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--module', type = str, help = 'path to inference module',
                        default = module_path)
    parser.add_argument('--dataset', type = str, help = 'path to tower dataset')
    parser.add_argument('--chains', type = int, default = 10,
                        help = 'Number of chains to run per trial')
    parser.add_argument('--steps', type = int, default = 3,
                        help = 'Number of inference steps')
    parser.add_argument('--slurm', action = 'store_true',
                        help = 'Use dask distributed on SLURM.')
    # pf parameters
    parser.add_argument('--mu', type = float, default = 3.0,
                        help = 'Mean for density prior')
    parser.add_argument('--sigma', type = float, default = 2.5,
                        help = 'Sigma for density prior')
    parser.add_argument('--width', type = float, default = 0.1,
                        help = 'Percent width for each particle')
    parser.add_argument('--bounds', type = float, nargs=2, default = (0.01, 13),
                        help = 'Lower and upper bounds for density')
    parser.add_argument('--particles', type = int, default = 4,
                        help = 'Number of particles per time step.')
    parser.add_argument('--resample', type = float, default = 0.5,
                        help = 'Probability that any given particle will be' +\
                        'resampled from the prior')
    parser.add_argument('--perturb', type = int, default = 1,
                        help = 'Number of perturbations per step')
    parser.add_argument('--pos_proposal', type = float, default = 0.1,
                        help = 'The width of the Uniform RV describing position')
    parser.add_argument('--pos_prior', type = float, default = 0.1,
                        help = 'The sigma of the Normal RV describing position')
    parser.add_argument('--trials', type = int, nargs = '+',
                        help = 'Specific trials to run.')
    parser.add_argument('--parameters', type = str, help = 'Path to '
                        'inference parameters. Used in distributed cases')

    args = parser.parse_args()
    # Some settings might be unintentionally overwritten
    use_slurm = args.slurm
    trials = args.trials
    # Check to see if we are branching off of a distributed call
    if args.parameters is None:
        print('Initializing new inference run')
        # assign unique name if new run
        out = os.path.basename(os.path.splitext(args.dataset)[0])
        out = os.path.join(CONFIG['PATHS', 'traces'], out + '_pf_results')
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out = out + '_' + suffix
        if not os.path.isdir(out):
            os.mkdir(out)
        # Save settings
        with open(os.path.join(out, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f, indent = 4)
    else:
        # Load the parameters from the json
        out = os.path.dirname(args.parameters)
        out = os.path.join(CONFIG['PATHS', 'traces'], out)
        param_file = os.path.join(out, 'parameters.json')
        print('Initializing from parameter file: ' + args.parameters)
        with open(param_file, 'r') as f:
            args = json.load(f)
            args = argparse.Namespace(**args)

    print('using slurm', use_slurm)
    print('Saving results in {0!s}'.format(out))

    # configure free parameters for inference
    parameters = load_default_parameters(args)

    dataset_path = os.path.join(CONFIG['PATHS', 'databases'],
                                           args.dataset)
    dataset = ParticleDataset(dataset_path)

    # figure out which trials are left to run
    if trials is None:
        trials = np.arange(len(dataset))

    # ignore trials that have all chains completed
    file_fn = lambda p: os.path.join(out, 'trial_{}.hdf5'.format(p))
    trials = list(filter(lambda t: not os.path.isfile(file_fn(t)),
                         trials))

    print('Number of trials remaining: {0:d}'.format(len(trials)))

    if use_slurm:
        # submit trials to slurm
        submit_sbatch(
            os.path.join(out, 'parameters.json'),
            trials,
        )
    else:
        # submit inference runs to dask
        tallies = {t_idx : 0 for t_idx in trials}
        for i,t_idx in enumerate(trials):
            if os.path.isfile(file_fn(t_idx)):
                print('Trial {0:d} already completed'.format(t_idx))
                continue
            else:
                print('Running trial {0:d}'.format(t_idx))
            priority = int(len(trials) - i)
            for chain in np.arange(args.chains):
                chain_path = chain_name(out, t_idx, chain)
                # check for chain save states
                if os.path.isfile(chain_path):
                    # update tally
                    print('Chain {} already completed'.format(chain_path))
                    tallies[t_idx] = tallies[t_idx] + 1
                else:
                    # Run chain
                    print('Running chain {}'.format(chain_path))
                    arguments = {
                        'dataset' : dataset_path,
                        'trial' : t_idx,
                        'out' : chain_path,
                        **parameters,
                    }
                    run_search(arguments)

            # chains completed
            out_path = os.path.join(out, 'trial_{}'.format(t_idx))
            chain_paths = list(map(lambda i: chain_name(out, t_idx, i),
                                       range(args.chains)))
            results = {}
            chains = np.empty((args.chains,), dtype = object)
            for c in range(args.chains):
                c_path = chain_paths[c]
                #load necessary bits
                with open(c_path, 'r') as f:
                    d = json.load(f)
                chains[c] = {k:d[k] for k in ('estimates', 'scores')}
                if c == 0:
                    results['gt'] = d['gt']
                    results['xs'] = d['xs']
                # cleanup chains
                os.remove(c_path)

            results['chains'] = dict(enumerate(chains))
            out_f = hdf5.HDF5(out_path + '.hdf5')
            out_f.write(results)



###############################################################################
# DASK Helpers
###############################################################################


def submit_sbatch(src, trials, size = 1000):

    chunks = min(size, len(trials))

    interpreter = '#!/bin/bash'
    extras = [
        # 'source /etc/profile.d/modules.sh',
        # 'module add openmind/singularity/3.0',
        # 'export PYTHONPATH="{0!s}"'.format(CONFIG['PATHS', 'root']),
    ]
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
    flags = ('--parameters {0!s}'.format(src),)
    jobs = [('--trials {0:d}'.format(p),) for p in trials]
    path = os.path.realpath(__file__)
    func = 'cd {0!s} && '.format(CONFIG['PATHS', 'root']) +\
           './run.sh python3 -W ignore {0!s}'.format(path)
    batch = sbatch.Batch(interpreter, func, jobs, flags, extras,
                         resources)
    print("Template Job:")
    print('\n'.join(batch.job_file(chunk=chunks)))
    batch.run(n = chunks, check_submission = False)

def initialize_dask(n):
    """ Setups up dask client and cluster

    Arguments:
        n (int): The number of tasks expected to run
        slurm (optional, bool): If true, use `SlurmCluster`
    """
    cores =  len(os.sched_getaffinity(0))
    nw= 1 # min(int(n), cores)
    print('Creating distributed client with {0:d} workers'.format(nw))
    cluster = distributed.LocalCluster(
        processes = False,
        n_workers = nw,
        threads_per_worker = 1)
    print(cluster.dashboard_link)
    return distributed.Client(cluster)

if __name__ == '__main__':
    main()
