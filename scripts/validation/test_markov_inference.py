#!/usr/bin/env python3
"""
Verifies that inference using MC simulator converges properly
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
module_path = os.path.join(root, 'inference',
                           'mc_physics_validation.jl')
inference = initialize(module_path)

def run_search(scene_json, scene_pos, out):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    positions = np.load(scene_pos)
    with open(scene_json, 'r') as f:
        scene_data = json.load(f)['scene']

    # runs inference and stores results
    results = inference(scene_data, positions, out)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trial', type = str, help = 'path to scene file')
    # misc
    parser.add_argument('--out', type = str, help = 'directory to save traces',
                        default =  'match_legacy')
    parser.add_argument('--fresh', action = 'store_true',
                        help = 'Ignore previous profiles')

    args = parser.parse_args()

    print('Initializing inference run')
    if args.fresh:
        print('Running funky fresh.. Previous saves will be overwritten')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = os.path.basename(os.path.splitext(args.trial)[0])
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
