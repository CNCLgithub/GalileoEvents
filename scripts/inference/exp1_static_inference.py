#!/usr/bin/env python3
"""
Computes MH-MCMC over Exp1 scenes
"""
import os
import glob
import json
import argparse
import datetime
import numpy as np

from galileo_ramp.utils import config
from galileo_ramp.utils.dataset.exp1_dataset import Exp1Dataset
from galileo_ramp.inference.execute import initialize

CONFIG = config.Config()

root = CONFIG['PATHS', 'root']
module_path = os.path.join(root, 'inference',
                           'queries', 'exp1_static_inference.jl')
# inference = initialize(module_path)

def run_search(scene_data, obs, time_points, out, iterations):
    """Runs a particle filter over the tower designated by the trial index

    Arguments:
        trial_idx (int): The index to access the trial.
        dataset_path (str): Path to the dataset.
        parameters (dict, optional): A dictionary containing parameters for search.

    Returns:
        A `dict` containing the inference trace.
    """
    for i,t in enumerate(time_points):
        out_path = "{0!s}_t_{1:d}".format(out, i)
        inference(scene_data, obs[:t, :, :], out_path, iterations)


def main():

    parser = argparse.ArgumentParser(
        description = 'Performs a particle-filter search over' + \
        'the galileo ball-ramp-world.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trial', type = int, help = 'index of scene file')
    parser.add_argument('--chains', type = int, default = 1,
                        help = 'number of chains')
    parser.add_argument('--dataset', type = str,
                        default = 'ramp_experiment_fixed_camera_fixed_ground_2018-04-30.hdf5',
                        help = 'path to scene dataset')
    parser.add_argument('--iterations', type = int,
                        default = 1000,
                        help = 'Number of iterations')

    # misc
    parser.add_argument('--out', type = str, help = 'directory to save traces',
                        default =  'exp1_static_inference')

    args = parser.parse_args()

    print('Initializing inference run')
    # assign unique name if new run
    out = os.path.join(CONFIG['PATHS', 'traces'], args.out)
    trial_name = 'trial_{0:d}'.format(args.trial)
    scene_json = os.path.join(CONFIG['PATHS', 'scenes'], 'legacy_converted',
                              trial_name + ".json")
    if not os.path.isdir(out):
        try:
            os.mkdir(out)
        except:
            print('{0!s} already exists'.format(out))

    dataset_file = os.path.join(CONFIG['PATHS', 'scenes'], args.dataset)
    dataset = Exp1Dataset(dataset_file)
    positions, time_points = dataset[args.trial]
    positions = np.transpose(positions, (1, 0, 2))

    # get proper time points
    if (args.trial < 120) and (args.trial % 2 == 1):
        _, time_points = dataset[args.trial - 1]

    with open(scene_json, 'r') as f:
        scene_data = json.load(f)['scene']

    print('Saving results in {0!s}'.format(out))
    for c in range(args.chains):
        out_path = os.path.join(out, trial_name)
        out_path = "{0!s}_chain_{1:d}".format(out_path, c)
        if os.path.isfile(out_path + '_trace.csv'):
            print('Inference already complete')
        else:
            run_search(scene_data, positions, time_points,
                       out_path,
                       args.iterations)


if __name__ == '__main__':
    main()
