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
module_path = os.path.join(root, 'inference', 'mh.jl')

def run_search(scene_args, dist_args, inf_args, out):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    smc = initialize(module_path)
    print('RUNNING SMC')
    results = smc(scene_args, dist_args, inf_args)
    # dict containing: gt, xs, scores, estimates
    pprint(results)
    with open(out, 'w') as f:
        json.dump(results, f, cls = encoders.NpEncoder)

###############################################################################
# Helpers
###############################################################################

def format_parameters(args, trial):
    """ Formats inputs from `argparse.ArgumentParser` for `smc.SMC`.

    Arguments:
        args: A `Namespace` collection

    Returns:
        A dictionary of relevant variables for inference.
    """

    with open(trial, 'r') as f:
        data = json.load(f)

    balls = sorted(list(data['scene']['objects'].keys()))
    ts = np.array([10, 100])
    d = {
        'scene_args' : [data['scene'], balls, ['density'], ts],
        'inf_args': [args.steps, args.factorize],
        'dist_args': {
            'prior' : np.array([args.bounds,]),
            'prop'  : np.array([[args.width, *args.bounds]]),
        },
    }
    return d

def load_params(path):
    param_path = os.path.join(CONFIG['PATHS', 'traces'], path)
    with open(param_path, 'r') as f:
        params = json.load(f)
    return argparse.Namespace(**params)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trial', type = str, help = 'path to scene file')
    parser.add_argument('params', type = load_params,
                        help = 'path to parameter json')
    # misc
    parser.add_argument('--out', type = str, help = 'directory to save traces',
                        default =  'pf_test')
    parser.add_argument('--fresh', action = 'store_true',
                        help = 'Ignore previous profiles')

    args = parser.parse_args()

    print('Initializing inference run')

    ratio = args.trial.split(os.sep)[-2]
    if args.fresh:
        print('Running funky fresh.. Previous saves will be overwritten')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = os.path.basename(os.path.splitext(args.trial)[0])
    trial_name = ratio + '-' + trial_name
    if not os.path.isdir(out):
        os.mkdir(out)

    print('Saving results in {0!s}'.format(out))

    # ignore trials that have all chains completed
    trial_fn = os.path.join(out, (trial_name + '.hdf5'))
    if os.path.isfile(trial_fn):
        print('Trial already completed at: ' + trial_name)
        if args.fresh:
            print('Overwriting: {0!s}'.format(trial_fn))
            os.remove(trial_fn)
        else:
            print('Exiting')
            return

    print('Running trial: ' + trial_name)
    chain_str = os.path.join(out, trial_name + '_chain_{}')
    chain_paths = list(map(lambda c: chain_str.format(c),
                       range(args.params.chains)))
    parameters = format_parameters(args.params, args.trial)

    param_fn = os.path.join(out, (trial_name + '.json'))


    for chain in chain_paths:
        # check for chain save states
        if os.path.isfile(chain):
            print('Chain {} already completed'.format(chain))
            if args.fresh:
                print('Overwriting: {0!s}'.format(chain))
                os.remove(chain)
            else:
                continue

        # Run chain
        print('Running chain {}'.format(chain))
        arguments = {'out' : chain, **parameters}
        run_search(**arguments)

    # Aggregate chains
    results = {}
    chains = np.empty((args.params.chains,), dtype = object)
    for c in range(args.params.chains):
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
