""" Evaluates the convergence statistics across several free parameter settings

Given a collection of free parameters, determines the probability
that the generated chains result in a ground truth mass ratio judgment.

"""


import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

from utils import config
from datasets.particle_dataset import ParticleDataset

CONFIG = config.Config()


def path_to_trial(path, dataset):
    """ Returns the trial and condition index for a path in condlist """
    path = os.path.splitext(path)[0]
    parts = path.split('_')
    trial_idx = int(parts[1])
    cond = int(parts[-1])
    (obs, b_id), tower = dataset[trial_idx]
    density = tower.graph.nodes[int(b_id)]['substance']['density']
    return (trial_idx, cond, density)

def main():

    parser = argparse.ArgumentParser(
        description = 'Verifies the balancing in a condlist file.')
    parser.add_argument('condlist', type = str,
                        help = 'Paths to the condlist')
    parser.add_argument('dataset', type = str,
                        help = 'Paths to dataset used to generate condlist')
    args = parser.parse_args()

    with open(args.condlist, 'r') as f:
        conditions = json.load(f)
    n_conds = len(conditions)
    dataset = ParticleDataset(args.dataset)
    dataframe = pd.DataFrame()
    columns = ('trial', 'time', 'mass') # relevant for histograms
    for c_i, trial_list in enumerate(conditions):
        c_data = zip(*map(lambda x: path_to_trial(x, dataset),
                          trial_list))
        c_data = dict(zip(columns, c_data))
        c_data = pd.DataFrame(c_data)
        c_data['condition'] = c_i
        dataframe = dataframe.append(c_data)

    fig, axs = plt.subplots(len(columns), n_conds,
                            figsize = (8 * n_conds, len(columns)*8))
    for cond, group in dataframe.groupby('condition'):
        print('Verifying condition {0:d}'.format(cond))
        for row, metric in enumerate(columns):
            ax = axs[row, cond]
            ax.set_title('Histogram for {0!s}, condition {1:d}'.format(metric,
                                                                       cond))
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            label = 'condition: {0:d}'.format(cond)
            data = group[metric].value_counts()
            ax.bar(data.axes[0], data, label = label)
            if metric != 'trial':
                print(data)

    fig.savefig(
        os.path.join(os.path.dirname(args.condlist), 'condlist-verify.png'),
        bbox_inches = 'tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
