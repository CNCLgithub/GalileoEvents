#!/usr/bin/env python3

import os
import json
import shlex
import argparse
import subprocess
import numpy as np
from pprint import pprint

from utils import config
CONFIG = config.Config()

def ffmpeg(source, out, vframes = None, extend = 0, image = None, fps = 60):
    """ Combines a series of rendered frames into mp4s

    Arguments:
    - source  (str): A string expression for ffmpeg to match frames
    - out     (str): The path to save the mp4
    - vframes (int, optional): The cutoff frame
    - extend  (int, optional): The number of frames to extend the last frame
    - image   (str, optional): A mask to append to the last frame
    - fps     (int, optional): Frames per second for mp4

    Returns:
    None
    """

    cmd = 'ffmpeg -y -framerate 60 -i {1!s} -hide_banner -crf 5 -preset slow -c:v libx264  -pix_fmt yuv420p'
    cmd = cmd.format(fps, source)
    if not vframes is None:
        cmd += ' -vframes {0:d}'.format(vframes)
    cmd += ' ' + out
    print('CMD', cmd)
    subprocess.run(shlex.split(cmd), check=False)
    if extend > 0 :
        if not os.path.isfile(out):
            print('Missing', out)
            return
        tsrc = out.replace('.mp4', '_t.mp4')
        os.rename(out, tsrc)
        cmd = ('ffmpeg -i {0!s} -filter_complex ' +\
               '\"[0]trim=0:2[a];[0]setpts=PTS-2/TB[b];[0][b]overlay[c];[a][c]concat\"' + \
               ' {1!s} -y').format(tsrc, out)
        subprocess.run(shlex.split(cmd), check=False)
        os.remove(tsrc)
    # if not image is None:
    #     tsrc = out + '.t'
    #     os.rename(out, tsrc)
    #     cmd = 'ffmpeg -y -i {0!s} -loop 1 -t 1 -i {1!s} -f lavfi -t 1 -i ' +\
    #           'anullsrc -filter_complex \"[1:v]scale=WxH:force_original_a' +\
    #           'spect_ratio=decrease,pad=W:H:\'(ow-iw)/2\':\'(oh-ih)/2\'[i]'+\
    #           ';[0:v][0:a][i][2:a] concat=n=2:v=1:a=1[v][a] \" -c:v libx26'+\
    #           '4 -c:a aac -map\" [v] \"-map\" [a]\" {2!s}'
    #     cmd = cmd.format(tsrc, image, out)
    #     print('CMD', cmd)
    #     subprocess.run(shlex.split(cmd), check=False)
    #     os.remove(tsrc)

def ffmpeg_concat(src1, src2, base = 1):
    """ Concatenates two mp4s

    Generates a new mp4 with the same name as base

    Arguments:
    - src1 (str): Path to first video
    - src2 (str): Path to second video.
    - base (int, optional): Which of the two sources to overwrite (0,1).

    Returns:
    None
    """
    if not os.path.isfile(src1):
        print('Missing', src1)
        return
    if not os.path.isfile(src2):
        print('Missing', src2)
        return
    base_src = (src1, src2)[base]
    tsrc = base_src.replace('.mp4', '_t.mp4')
    os.rename(base_src, tsrc)
    sources = [src1, src2]
    sources[base] = tsrc
    # cmd = 'ffmpeg -y \"concat:{0!s}|{1!s}\" -c copy {2!s}'
    cmd = 'ffmpeg -y -f concat -safe 0 -i <(for f in {0!s} {1!s}; do echo '+\
          '\"file \'$f\'\"; done) -c copy {2!s}'
#     cmd = 'ffmpeg -i {0!s} -i {1!s} \
# -filter_complex \"[0:v:0][1:v:0]concat=n=2:v=1[outv]\" \
# -map \"[outv]\" {2!s}'
    cmd = cmd.format(*sources, base_src)
    print('CMD', cmd)
    subprocess.run(cmd, check = False, shell = True, executable="/bin/bash")
    os.remove(tsrc)

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates stimuli based off inference timings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('run', type = str,
                        help = 'Paths to the PF runs')
    parser.add_argument('renders', type = str,
                        help = 'Paths to the trial renderings')
    parser.add_argument('--panorama', type = int, default = 0,
                        help = 'Add panorama to beginning of the trial '
                        'with the given fps. 0 will result in no action')
    parser.add_argument('--mask', action = 'store_true',
                        help = 'Add mask to non-terminal conditions')
    args = parser.parse_args()

    workdir = os.path.join(CONFIG['PATHS', 'traces'], args.run)
    render_src = os.path.join(CONFIG['PATHS', 'renders'], args.renders)
    # Load timings
    timing_file = os.path.join(workdir, 'time_points.json')
    with open(timing_file, 'r') as f:
        timings = json.load(f)

    # Set mask
    if args.mask:
        mask = os.path.join(CONFIG['PATHS', 'root'], 'experiment', 'render',
                            'Textures', 'mask.mp4')
    else:
        mask = None

    # Movies will be saved within the inference directory
    movie_dir = os.path.join(workdir, 'movies_panorama_{0:d}_mask_{1:d}')
    movie_dir = movie_dir.format(args.panorama, args.mask)
    if not os.path.isdir(movie_dir):
        os.mkdir(movie_dir)

    trial_keys = list(timings.keys())
    for trial in trial_keys:
        if args.panorama > 0:
            # create panorama
            src_path = '{0!s}/render/frozen/%d.png'.format(trial)
            src_path = os.path.join(render_src, src_path)
            pan_path = 'trial_{0!s}_pan.mp4'.format(trial)
            pan_path = os.path.join(movie_dir, pan_path)
            ffmpeg(src_path, pan_path, fps = args.panorama, vframes = 120,
                   extend = 1)

        # Create motion component
        src_path = '{0!s}/render/motion/%d.png'.format(trial)
        src_path = os.path.join(render_src, src_path)

        t_idx = int(trial)
        if t_idx % 2 == 0:
            t_idx += 1
        times = timings[str(t_idx)]

        for cond, point in enumerate(times):
            out_path = 'trial_{0!s}_cond_{1:d}.mp4'.format(trial, cond)
            out_path = os.path.join(movie_dir, out_path)

            # Create raw video
            ffmpeg(src_path, out_path, vframes = point)
            if not mask is None:
                ffmpeg_concat(out_path, mask, base = 0)

            # concatenate the panorama and motion videos
            if args.panorama > 0:
                ffmpeg_concat(pan_path, out_path, base = 1)



if __name__ == '__main__':
    main()