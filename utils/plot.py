import os
from pprint import pprint

import h5py
import numpy as np
from scipy import stats

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

def plot_direction_diff(data, quantiles, radius, out):

    cmap = plt.get_cmap('jet_r')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                      projection='polar') #, facecolor='#d5de9c')
    ax.set_title('Angle differentials.')
    ax.set_rmax(radius)
    ax.set_aspect('equal', adjustable='box')
    ax.grid = True
    for idx, q in enumerate(quantiles):
        mut, ori = data[idx]
        color = cmap(float(idx)/len(quantiles))
        ax.plot([0, ori.angle], [0, ori.mag], c = color,
                label = '{0:d}%'.format(int(q * 100)))
        ax.plot([0, mut.angle], [0, mut.mag], c = color,
                ls = '--')

    ax.legend()
    fig.savefig(out, bbox_inches = 'tight')

def plot_tower_diff(df, out, metric):
    """ Plots a 2-d histogram of tower differentials (stability, direction).
    """
    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    ax.set_xlabel(metric)
    # ax.set_ylabel('Angle (Radians)')
    ax.set_ylabel('L2 Distance')
    ax.set_title('Physics differentials')

    xs = df[metric]
    ys = df['l2']
    # print(xs.describe())
    # print(ys.describe())
    # print(np.mean(ys))
    h = ax.hist2d(xs, ys, cmap = cm.hot,
                  # bins = 25,
                  normed = False)
    fig.colorbar(h[3], ax = ax)
    fig.savefig(out, bbox_inches = 'tight')


def plot_pf_trace(trace, out):

    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    ax.set_xlabel('Frames')
    # xs = np.arange(len(trace['map_latent']))
    block = list(trace[0]['gt'].keys())[0]
    gt = trace[0]['gt'][block]['density']
    ax.axhline(gt, color = 'g', label = 'GT', linestyle = '--')

    for t in trace:
        ax.plot(t['xs'], t['map_latents'])
        ax.set_ylabel('Density', color = 'r')
        ax.set_ylim([0, 13])

    fig.savefig(out, bbox_inches='tight')

    # pprint(trace)
    # xs = np.arange(len(trace['map_latent']))
    # block = list(trace['gt'].keys())[0]
    # gt = trace['gt'][block]['density']
    # ax.axhline(gt, color = 'g', label = 'GT', linestyle = '--')
    # ax.axvline(trace['max_delta_map'], label = 'Max Delta MAP')
    # ax.axvline(trace['init_delta_map'], label = 'Delta MAP')

    # ax.scatter(xs, trace['map_latent'], label = 'map_latent',
    #            color = 'r')
    # ax.set_ylabel('Density', color = 'r')
    # ax.set_ylim([0, 13])
    # ax.legend()

    # ax2 = ax.twinx()
    # ax2.set_ylabel('MAP', color = 'b')
    # ax2.set_ylim([0, 1])
    # ax2.plot(xs, trace['map'], color = 'b')
    # ax2.plot(xs, trace['map_l'], color = 'y')
    # fig.savefig(out, bbox_inches='tight')

def plot_run(path, gt, obs, out):
    """
    """
    n_latents = len(gt)
    latents = list(gt.keys())
    components = gt[latents[0]]
    n_components = len(components)
    with h5py.File(path, 'r') as f:
        n_frames = len(f)
        data = {l : [] for l in latents}
        data.update({'posteriors' : []})

        for i in range(n_frames):
            i_key = '{0:d}'.format(i)
            iteration = f[i_key]
            n_samples = len(iteration['trace'])
            guesses = [iteration['trace']['{}'.format(i)]['latents']
                       for i in range(n_samples)]
            zs = iteration['posteriors']
            if i == 0:
                data['posteriors'] = zs
            else:
                data['posteriors'] = np.vstack((data['posteriors'], zs))

            for col, latent in enumerate(latents):
                vals = np.array([[g[latent][c].value for c in components]
                                 for g in guesses])
                vals = np.expand_dims(vals, 0)
                if i == 0:
                    data[latent] = vals
                else:
                    data[latent] = np.vstack((data[latent], vals))


        fig, axs = plt.subplots(max(2, len(latents)),  n_components,
                                figsize = (8, 8))

        post_line = {}
        for c, latent in enumerate(latents):
            for c2, comp in enumerate(components):
                ax = axs[c]
                # ax = axs
                ax.set_xlim([0, n_frames])
                ax.set_ylim([0, 10])
                ax.set_xlabel('Frames')
                ax.set_ylabel(comp)

                if c2 == 0:
                    ax.set_title(latent)

                xs = np.repeat(np.arange(n_frames), n_samples)
                ys = data[latent][:, :, c2].flatten()

                ax.hist2d(xs, ys,
                          range = [(0, n_frames), (0, 10)],
                          cmap = cm.hot,
                )

                ax.axhline(gt[latent][comp], color = 'g')

                best_i = np.argmax(data['posteriors'], axis = 1)
                best_i = np.array(list(enumerate(best_i)))
                best = data[latent][:, :, c2][best_i[:,0], best_i[:,1]]
                ax.scatter(np.arange(n_frames), best, color = 'b')
                post_line[latent] = best

        # ax = axs[-1]
        # ax.set_xlabel('Frames')
        # ax.set_ylabel('Obs')
        # ax.plot(np.arange(n_frames + 1), obs)
        # points = [obs_f(post_line['m'][i], post_line['b'][i], i+1)[i]
        #           for i in range(n_frames)]
        # points = np.array(points).flatten()
        # ax.plot(np.arange(n_frames) + 1, points)

    fig.savefig(out, bbox_inches='tight')
