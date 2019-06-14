import os
import sys
import bpy
import json
import time
import argparse
import mathutils

import numpy as np
#################################################
# https://stackoverflow.com/questions/28075599/opening-blend-files-using-blenders-python-api
from bpy.app.handlers import persistent
@persistent
def load_handler(dummy):
    print("Load Handler:", bpy.data.filepath)
bpy.app.handlers.load_post.append(load_handler)
#################################################


class RampScene:

    '''
    Interface for bpy.
    '''

    def __init__(self, scene_json, materials_path, trace = None,
                 wire_frame = False, theta = None):
        """
        Arguments:
            scene_json (str): A serialized tower string.
            materials_path (str): Path to a .blend file containing materials.
            trace (optional, list): A list of traces, [sim1, sim2, ...]
                 where each sim is a complete trace for all blocks.
            theta (optional, float): An angle to orient the camera.
            wireframe : DEPRECATED
        """

        # Initialize attributes
        self.trace = trace
        self.theta = theta

        # Clear scene
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete(use_global=False)
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)

        # Load materials and textures
        with Suppressor():
            bpy.ops.wm.open_mainfile(filepath=materials_path)

        # Parse scene structure
        self.load_scene(scene_json)

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, t):
        if not t is None:
            frames = len(t['pos'])
        else:
            frames = 1
        bpy.context.scene.frame_set(1)
        bpy.context.scene.frame_end = frames + 1
        self._trace = t

    def select_obj(self, obj):
        """
        Brings the given object into active context.
        """
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.context.scene.objects.active
        bpy.context.scene.update()


    def rotate_obj(self, obj, rot):
        """
        Rotates the given object by the given quaternion.
        """
        self.select_obj(obj)
        if len(rot) == 3:
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = rot
        else:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = np.roll(rot, 1) # [3, 0, 1, 2]
        bpy.context.scene.update()

    def move_obj(self, obj, pos):
        """
        Moves the given object by the given 3-d vector.
        """
        self.select_obj(obj)
        pos = mathutils.Vector(pos)
        obj.location = pos
        bpy.context.scene.update()

    def scale_obj(self, obj, dims):
        """
        Rescales to the object to the given dimensions.
        """
        self.select_obj(obj)
        obj.dimensions = dims
        bpy.context.scene.update()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.context.scene.update()

    def set_appearance(self, obj, mat):
        """
        Assigns a material to a block.
        """
        if not mat in bpy.data.materials:
            raise ValueError('Unknown material {}'.format(mat))
        if mat == 'Wood':
            mat = 'rough_wood_{0:d}'.format(1)
        obj.active_material = bpy.data.materials[mat]
        bpy.context.scene.update()

    def create_block(self, name, object_d):
        """
        Initializes a ball.
        """
        bpy.ops.mesh.primitive_ico_sphere_add(location=object_d['position'],
                                              view_align=False,
                                              enter_editmode=False,
                                              subdivisions=7,
                                              size = object_d['dims'][0])
        ob = bpy.context.object
        ob.name = name
        ob.show_name = True
        me = ob.data
        me.name = '{0!s}_Mesh'.format(name)

        if 'appearance' in object_d:
            mat = object_d['appearance']
        else:
            mat = 'U'
        self.set_appearance(ob, mat)

    def load_scene(self, scene_dict):
        # Setup ramp
        ramp = bpy.data.objects['Ramp']
        ramp_d = scene_dict['ramp']
        self.move_obj(ramp, ramp_d['position'])
        self.rotate_obj(ramp, ramp_d['orientation'])
        # Load Objects
        for name, data in scene_dict['objects'].items():
            self.create_block(name, data)

    def set_rendering_params(self, resolution):
        """
        Configures various settings for rendering such as resolution.
        """
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.cycles.samples = 256
        bpy.context.scene.render.tile_x = 32
        bpy.context.scene.render.tile_y = 32
        bpy.context.scene.render.engine = 'CYCLES'

    def set_camera(self, rot):
        """
        Moves the camera along a circular path.
        Arguments:
            rot (float): angle in radians along path.
        """
        radius = 81.0
        # Move camera to position on ring
        xyz = [np.cos(rot) * radius, np.sin(rot) * radius, 35]
        camera = bpy.data.objects['Camera']
        camera.location = xyz
        bpy.context.scene.update()
        # The camera automatically tracks Empty
        camera_track = bpy.data.objects['Empty']
        self.move_obj(camera_track, [0, 0, 1])
        camera.keyframe_insert(data_path='location', index = -1)
        camera.keyframe_insert(data_path='rotation_quaternion', index = -1)


    def _frame_set(self,frame):
        """ Helper to `frame_set`.

        Arguments:
            sim (int) : index of trace.
            frame (int) : index for current frame.
        """
        positions = np.array(self.trace['pos'][frame])
        rotations = np.array(self.trace['orn'][frame])
        n_balls = len(positions)
        for ball_i in range(n_balls):
            ball = bpy.data.objects['{0:d}'.format(ball_i)]
            self.move_obj(ball, positions[ball_i])
            self.rotate_obj(ball, rotations[ball_i])
            ball.keyframe_insert(data_path='location', index = -1)
            ball.keyframe_insert(data_path='rotation_quaternion', index = -1)


    def frame_set(self, frame, rot):
        """ Updates the scene to the given frame.
        Arguments:
            frame (int): Index of the frame.
            rot (float): Rotation of camera.
        """
        if frame < 0:
            frame = len(self.trace[0]['position']) + frame
        bpy.context.scene.frame_set(frame)
        n_sims = len(self.trace)
        self._frame_set(frame)
        self.set_camera(rot)
        bpy.context.scene.update()


    def render(self, output_name, frames, show = [],
               resolution = (256, 256), camera_rot = None):
        """ Renders a scene.
        Arguments:
            output_name (str): Path to save frames
            frames (list) : a list of frames to render (shifted by warmup)
            show (list) : a list of object names to render
            resolution (tuple) : xy resolution.
            camerate_rot (0,360) : degrees around the scene to place the camera.
        """
        if not os.path.isdir(output_name):
            os.mkdir(output_name)
        self.set_rendering_params(resolution)
        if len(show) > 0:
            for obj in bpy.context.scene.objects:
                if not obj.name in show:
                    # print("Hiding {0!s}".format(o_name))
                    obj.cycles_visibility.diffuse = False
                    obj.hide = True
                    obj.hide_render = True

        if camera_rot is None:
            camera_rot = np.zeros(len(frames))
        for i, (frame, cam) in enumerate(zip(frames, camera_rot)):
            out = os.path.join(output_name, '{0:d}'.format(i))
            if os.path.isfile(out + '.png'):
                msg = 'Frame {} already rendered at {}'
                msg = msg.format(i, out)
                print(msg)
                continue
            bpy.context.scene.render.filepath = out
            self.frame_set(frame, cam)
            t_0 = time.time()
            with Suppressor():
                bpy.ops.render.render(write_still=True)
            dur = time.time() - t_0
            print('Rendering frame {} at {} took {}s'.format(i, out, dur))


    def render_circle(self, out_path, freeze = True, dur = 1,
                      resolution = (256, 256), theta = 0):
        """
        Renders a ring around a tower.
        Arguments:
            out_path (str): Path to save frames.
            freeze (bool): Whether or not to run physics.
            dur (float, optional): Duration in seconds.
            resolution (float, optional): Resolution of render.
        """
        self.set_rendering_params(resolution)
        n = int(dur * 60) # rendered at 60 fps
        rots = np.linspace(theta, theta + (np.pi * 2), n)
        if freeze == True:
            frames = np.zeros(n).astype(int)
        else:
            frames = np.arange(n)

        self.render(out_path, frames, resolution = resolution,
                    camera_rot = rots)

    def save(self, out, frames):
        """
        Writes the scene as a blend file.
        """
        for i in range(frames):
            self.frame_set(i, self.theta)
        bpy.ops.wm.save_as_mainfile(filepath=out)

# From https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class Suppressor(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def parser(args):

    p = argparse.ArgumentParser(description = 'Renders blockworld scene')
    p.add_argument('--scene', type = json.loads,
                   help = 'Tower json describing the scene.')
    p.add_argument('--trace', type = load_trace,
                   help = 'Trace json for physics.')
    p.add_argument('--materials', type = str,
                   help = 'Path to blender materials.')
    p.add_argument('--out', type = str,
                   help = 'Path to save rendering')
    p.add_argument('--save_world', action = 'store_true',
                   help = 'Save the resulting blend scene')
    p.add_argument('--render_mode', type = str, default = 'default',
                   choices = ['default', 'motion', 'frozen', 'none'],
                   help = 'mode to render')
    p.add_argument('--resolution', type = int, nargs = 2,
                   default = (256,256),  help = 'Render resolution')
    p.add_argument('--theta', type = float, default = 0,
                   help = 'Overrides camera angle. (radians)')
    p.add_argument('--gpu', action = 'store_true',
                   help = 'Use CUDA rendering')
    p.add_argument('--frames', type = int, nargs = '+',
                   help = 'Specific frames to render')
    return p.parse_args(args)


def load_trace(path):
    with open(path, 'r') as f:
        str = f.read()
        traces = json.loads(str)
    return traces

def main():
    argv = sys.argv
    print(argv[:6])
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser(argv)

    scene = RampScene(args.scene, args.materials, args.trace,
                       theta = args.theta)

    if args.gpu:
        print('Using gpu')
        bpy.context.scene.cycles.device = 'GPU'

    path = os.path.join(args.out, 'render')
    if not os.path.isdir(path):
        os.mkdir(path)

    frozen_path = os.path.join(path, 'frozen')
    motion_path = os.path.join(path, 'motion')
    if args.frames is None:
        n_frames = len(args.trace['pos'])
        frames = np.arange(n_frames)
    else:
        n_frames = len(args.frames)
        frames = args.frames

    if args.render_mode == 'default' or args.render_mode == 'frozen':
        scene.render_circle(frozen_path, freeze = True, dur = 2,
                            resolution = args.resolution, theta = args.theta)
    if args.render_mode == 'default' or args.render_mode == 'motion':
        scene.render(motion_path, frames,
                     camera_rot = np.repeat(args.theta, n_frames),
                     resolution = args.resolution,)

    if args.save_world:
        path = os.path.join(args.out, 'world.blend')
        n_frames = len(args.trace['pos'])
        scene.save(path, n_frames)

if __name__ == '__main__':
    main()
