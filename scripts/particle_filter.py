#!/usr/bin/env python3
"""
Simple test script that runs inference in Gen through python
"""
import os
import glob
import json
import argparse
import datetime
import numpy as np
from pprint import pprint

from galileo_ramp.utils import config, hdf5, encoders
from galileo_ramp.inference.execute import initialize

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']
module_path = os.path.join(root, 'inference', 'smc.jl')



def run_search(scene_args, dist_args, inf_args, inf_module, out):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    smc = initialize(inf_module)
    print('RUNNING SMC')
    results = smc(scene_args, dist_args, inf_args)
    # dict containing: gt, xs, scores, estimates
    pprint(results)
    with open(out, 'w') as f:
        json.dump(results, f, cls = encoders.NpEncoder)

###############################################################################
# Helpers
###############################################################################

def format_parameters(args):
    """ Formats inputs from `argparse.ArgumentParser` for `smc.SMC`.

    Arguments:
        args: A `Namespace` collection

    Returns:
        A dictionary of relevant variables for inference.
    """
    trial_path = os.path.join(CONFIG['PATHS', 'scenes'],
                              args.ratio,
                              args.trial)
    with open(trial_path, 'r') as f:
        data = json.load(f)

    balls = sorted(list(data['scene']['objects'].keys()))
    d = {
        'scene_args' : [data['scene'], balls, ['density'], 900],
        'inf_args': [args.particles, args.steps, args.resample, args.perturb],
        'dist_args': {
            'prior' : np.array([args.bounds,]),
            'prop'  : np.array([[args.width, *args.bounds]]),
        },
        'inf_module' : args.module
    }
    return d

def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('ratio', type = str, help = 'path to tower dataset')
    parser.add_argument('trial', type = str, help = 'path to tower dataset')
    parser.add_argument('--module', type = str, help = 'path to inference module',
                        default = module_path)
    parser.add_argument('--chains', type = int, default = 10,
                        help = 'Number of chains to run per trial')
    parser.add_argument('--steps', type = int, default = 10,
                        help = 'Number of inference steps')
    # pf parameters
    parser.add_argument('--mu', type = float, default = 1.0,
                        help = 'Mean for density prior')
    parser.add_argument('--sigma', type = float, default = 2.0,
                        help = 'Sigma for density prior')
    parser.add_argument('--width', type = float, default = 0.2,
                        help = 'Percent width for each particle')
    parser.add_argument('--bounds', type = float, nargs=2, default = (-2, 2),
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
    parser.add_argument('--out', type = str, help = 'directory to save traces',
                        default = os.path.join(CONFIG['PATHS', 'traces'], 'pf_test'))
    parser.add_argument('--fresh', action = 'store_true',
                        help = 'Ignore previous profiles')

    args = parser.parse_args()

    print('Initializing inference run')

    if args.fresh:
        print('Running funky fresh.. Previous saves will be overwritten')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = os.path.basename(os.path.splitext(args.trial)[0])
    trial_name = args.ratio + '-' + trial_name
    if not os.path.isdir(out):
        os.mkdir(out)

    print('Saving results in {0!s}'.format(out))

    # ignore trials that have all chains completed
    trial_fn = os.path.join(out, (trial_name + '.hdf5'))
    if os.path.isfile(trial_fn) and not args.fresh:
        print('Trial already completed at: ' + trial_name)
        return

    print('Running trial: ' + trial_name)
    chain_str = os.path.join(out, trial_name + '_chain_{}')
    chain_paths = list(map(lambda c: chain_str.format(c),
                       range(args.chains)))
    parameters = format_parameters(args)

    for chain in chain_paths:
        # check for chain save states
        if os.path.isfile(chain) and not args.fresh:
            # update tally
            print('Chain {} already completed'.format(chain))
            continue

        # Run chain
        print('Running chain {}'.format(chain))
        arguments = {'out' : chain, **parameters}
        run_search(**arguments)

    # Aggregate chains
    results = {}
    chains = np.empty((args.chains,), dtype = object)
    for c in range(args.chains):
        c_path = chain_paths[c]
        #load necessary bits
        with open(c_path, 'r') as f:
            d = json.load(f)
        chains[c] = {k:d[k] for k in ('estimates', 'scores')}
        if c == 0:
            results['xs'] = d['xs']
            results['latents'] = np.string_(d['latents'])
            results['gt'] = d['gt']

    results['chains'] = dict(enumerate(chains))
    out_f = hdf5.HDF5(trial_fn)
    out_f.write(results)

    # cleanup chains
    for c in chain_paths:
        os.remove(c)


if __name__ == '__main__':
    main()
