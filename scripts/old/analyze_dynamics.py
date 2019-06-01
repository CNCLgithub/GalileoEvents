#!/bin/python3
""" Computes physical statistics over block towers.
"""

import os
import glob
import copy
import json
import argparse
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

from datasets.descriptive_dataset import DescriptiveDataset
from utils import config
CONFIG = config.Config()


def main():
    parser = argparse.ArgumentParser(
        description = 'Renders the towers in a given directory')
    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to tower jsons')
    # parser.add_argument('--search', action = 'store_true',
    #                     help = 'Search through tower configurations.')

    args = parser.parse_args()

    dataset = DescriptiveDataset(args.src)
    for idx, (org, inc) in enumerate(dataset):
        if idx == 0:
            df = pd.DataFrame(columns = list(org.keys()) + ['scene'])
        df.loc[len(df)] = {'scene' : idx, **org}
        df.loc[len(df)] = {'scene' : idx, **inc}

    df = df[df['metric'] == 'instability_diff']
    fn = lambda r: 'orig' if r['id'] == 'template' else 'mut'
    df['mut_label'] = df.apply(fn, axis = 1)
    print(df)
    fig, axes = plt.subplots(2,1, figsize=(15,5))
    for (mut, m_group), ax in zip(df.groupby('mutation'), axes):

        groups = m_group.groupby('scene')
        df2 = pd.DataFrame(index = groups.groups.keys(),
                           columns = ['orig', 'mut'])
        for scene, group in groups:
            df2.loc[scene, group['mut_label']] = list(group['instability'])

        print(df2)
        ax = df2.plot.bar(ax = ax, title = mut)

    out = os.path.join(CONFIG['PATHS', 'renders'],
                       os.path.basename(args.src).replace('hdf5', 'png'))
    fig.savefig(out)
    
    # print(df.groupby('scene').apply(fn))






    # b_width = 0.25 # for row, ys in enumerate(stabilities):
    # ax = axes[row]
    # xs = np.arange(len(ys)*row, len(ys)*(row + 1))
    # ax.bar(xs, ys[:, 0], color ='C0')
    #     ax.bar(xs + b_width, ys[:, 1], color = 'C1')
    #     ax.set_title(['Light', 'Heavy'][row])
    #     ax.set_xlabel('Scene')
    #     ax.set_ylabel('Instability')
    #     ax.set_ylimit([0, 1.0])

    

if __name__ == '__main__':
    main()

