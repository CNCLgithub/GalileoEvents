import os
import shlex
import argparse
import subprocess

import numpy as np

from config import Config
CONFIG = Config()

def ffmpeg(source, out, extend = 0, image = None):
    cmd = ("ffmpeg -r 30 -i {0!s} -pix_fmt yuv420p -vcodec libx264 "+\
           "{1!s}").format(source, out)
    subprocess.run(shlex.split(cmd))
    if extend > 0:
        cmd = ('ffmpeg -i {0!s} -filter_complex ' +\
               '\"[0]trim=0:2[a];[0]setpts=PTS-2/TB[b];[0][b]overlay[c];[a][c]concat\"' + \
               ' {0!s} -y').format(out)
        subprocess.run(shlex.split(cmd))



def main():
    parser = argparse.ArgumentParser(
        description = "Generates movie from scene")

    parser.add_argument('--src', type = str, default = 'towers',
                        help = 'Path to rendered frames')

    args = parser.parse_args()
    # src = os.path.join(CONFIG['data'], args.src + '_diff_render')
    src = os.path.join(CONFIG['data'], args.src + '_render')

    towers = next(os.walk(src))[1]

    for tower in towers:
        fp = '{0!s}_motion.mp4'.format(tower)
        fp = os.path.join(src, fp)
        path_str = os.path.join(src, tower, 'render', 'motion',
                                '%d.png')
        ffmpeg(path_str, fp)
    # quantiles = next(os.walk(src))[1]
    # for quant in quantiles:
    #     towers = next(os.walk(os.path.join(src, quant)))[1]

    #     for tower in towers:
    #         fp = '{0!s}_{1!s}_motion.mp4'.format(quant, tower)
    #         fp = os.path.join(src, fp)
    #         path_str = os.path.join(src, quant, tower, 'render', 'motion',
    #                                 '%d.png')
    #         ffmpeg(path_str, fp)

if __name__ == '__main__':
    main()
