#!/usr/bin/env python3
"""
Attempts to find physical settings that matches
the current forward model to traces in the legacy model.
"""
import os
import glob
import json
import argparse
import datetime
import numpy as np

from galileo_ramp.utils import config
from galileo_ramp.inference.execute import initialize

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']
module_path = os.path.join(root, 'inference',
                           'queries', 'match_legacy_physics.jl')
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
    positions = np.transpose(positions, (1, 0, 2))
    with open(scene_json, 'r') as f:
        scene_data = json.load(f)['scene']

    inference(scene_data, positions, out)


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

    args = parser.parse_args()

    print('Initializing inference run')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = os.path.basename(os.path.splitext(args.trial)[0])
    if not os.path.isdir(out):
        os.mkdir(out)

    out_path = os.path.join(out, trial_name)
    print('Saving results in {0!s}'.format(out))
    position_file = args.trial.replace('.json', '_pos.npy')
    run_search(args.trial, position_file, out_path)


if __name__ == '__main__':
    main()
