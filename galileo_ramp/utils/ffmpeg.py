import os
import json
import shlex
import subprocess
import numpy as np
from pprint import pprint

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
    cmd = cmd.format(*sources, base_src)
    print('CMD', cmd)
    subprocess.run(cmd, check = False, shell = True, executable="/bin/bash")
    os.remove(tsrc)
