"""Creates a directory tree containing trials."""

import os
import glob
import json
import shutil
import argparse
import datetime
import numpy as np
from pprint import pprint

from utils import config

CONFIG = config.Config()


def retrieve_towers(source, n):
    """Selects viable towers, balancing across heavy and light.

    Arguments:
        source (str): Path to directory containing towers
        n      (int): Number of heavy, light tower pairs
    """
    file_list = glob.glob(os.path.join(source, '*.json'))
    paths = file_list[:n]
    return paths

def write_trial(tower_path, out, imgs = None, description = None):
    """Writes a given trial info to disk

    Arguments:
        tower_path (str): Path to tower scene
        out (str): Path to trial directory

        imgs (list, optional): Paths to rendered frames
        description (dict, optional): Extra info on trial
    """
    if not os.path.isdir(out):
        os.mkdir(out)

    # shutil.copyfile(tower_path, os.path.join(out, 'tower'))
    # Split up the json for read speed in hdf5
    with open(tower_path, 'r') as f:
        tower_data = json.load(f)
    for key in tower_data:
        with open(os.path.join(out, key), 'w') as f:
            json.dump(tower_data[key], f)

    if not imgs is None:
        for img_idx,img in enumerate(imgs):
            f = os.path.join(out, '{0:d}.png'.format(img_idx))
            shutil.copyfile(img, f)

    if not description is None:
        with open(os.path.join(out, 'description'), 'w') as f:
            json.dump(description, f)

def generate_trials(paths, out, mode):
    """Stores trial data simulated from tower descriptions

    Arguments:
        paths (list): Paths to tower scenes
        out   (str) : Path to directory tree
        mode  (str) : Type of dataset to create
    """
    if not os.path.isdir(out):
        os.mkdir(out)
    out = os.path.join(out, 'trials')
    if not os.path.isdir(out):
        os.mkdir(out)


    for idx, scene in enumerate(paths):
        out_path = os.path.join(out, '{0:d}'.format(idx))
        description = {
            'hash' : os.path.basename(scene)[:-5]
        }
        write_trial(scene, out_path, description = description)

def main():
    parser = argparse.ArgumentParser(
        description = 'Creates tower datasets')

    parser.add_argument('sources', type = str, nargs = '+',
                        help = 'Paths to tower jsons')
    parser.add_argument('--mode', type = str, choices = ['particle_filter'],
                        default = 'particle_filter',
                        help = 'Type of dataset to generate')
    parser.add_argument('--number', type = int, default = 30,
                        help = 'Number of heavy light pairs.')
    parser.add_argument('--out', type = str,
                        help = 'Path to save dataset.')
    args = parser.parse_args()

    if not args.out is None:
        out = args.out
    else:
        suffix = datetime.datetime.now().strftime("%m_%d_%y_%H%M%S")
        out = "_".join([args.mode, suffix])
        out = os.path.join(CONFIG['PATHS', 'databases'], out)

    trials = []
    for source in args.sources:
        paths = retrieve_towers(os.path.join(CONFIG['PATHS', 'towers'], source),
                                args.number)
        trials.extend(paths)

    print('Saving to {}'.format(out))
    generate_trials(trials, out, args.mode)

    info = {
        'trials' : len(trials),
        'sources' : args.sources,
        'time' : datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    }
    info_out = os.path.join(out, 'info')
    with open(info_out, 'w') as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
