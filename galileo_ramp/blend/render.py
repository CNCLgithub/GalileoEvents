""" BPY script that handles rendering logic for ramp world
"""
import os
import sys
try:
    import bpy
    import mathutils
except:
    # For documentation
    print('No `bpy` available')
import json
import time
import argparse

import numpy as np

# Flush stdout in case blender is complaining
sys.stdout.flush()

class RampScene:

    """
    Defines the ramp world in bpy.
    """

    def __init__(self, scene, trace = None, theta = None):
        """ Initializes objects, physics, and camera

        :param scene: Describes the ramp, table, and balls.
        :type scene_d: dict
        :param trace: the physical state of the objects
        :type trace: dict or None
        :param theta: Angle around the world to point the camera
        :type theta: float or None
        """
        # Initialize attributes
        self.trace = trace
        self.theta = theta

        # Parse scene structure
        self.load_scene(scene)
        print('Loaded scene')
        sys.stdout.flush()

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, t):
        """
        A dictionary containing to keys: `('pos', 'rot')` where each
        holds a 3-dimensional array of `TxNxK`,
        where `T` is the number of key frames,
        `N` is the number of objects,
        and `K` is either `xyz` or `wxyz`.

        :param t: The physics state to apply as keyframes
        :type t: dict or None
        """
        if not t is None:
            frames = len(t['pos'])
        else:
            frames = 1
        bpy.context.scene.frame_set(1)
        bpy.context.scene.frame_end = frames + 1
        self._trace = t

    def select_obj(self, obj):
        """ Sets the given object into active context.
        """
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.update()


    def rotate_obj(self, obj, rot):
        """ Rotates the object.

        :param rot: Either an euler angle (xyz) or quaternion (wxyz)
        """
        self.select_obj(obj)
        if len(rot) == 3:
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = rot
        else:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = np.roll(rot, 1) # [3, 0, 1, 2]
        bpy.context.view_layer.update()

    def move_obj(self, obj, pos):
        """ Moves the object.

        :param pos: An xyz designating the object's new location.
        """
        self.select_obj(obj)
        pos = mathutils.Vector(pos)
        obj.location = pos
        bpy.context.view_layer.update()

    def scale_obj(self, obj, dims):
        """ Rescales to the object to the given dimensions.
        """
        self.select_obj(obj)
        obj.dimensions = dims
        bpy.context.view_layer.update()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.context.view_layer.update()

    def set_appearance(self, obj, mat):
        """ Assigns a material to a block.

        If the material is not defined within the blend file,
        the object is assigned `"Wood"`.
        """
        # if not mat in bpy.data.materials:
        #     raise ValueError('Unknown material {}'.format(mat))
        if mat == 'Wood':
            mat = 'rough_wood_{0:d}'.format(1)
        if mat in bpy.data.materials:
            obj.active_material = bpy.data.materials[mat]
        bpy.context.view_layer.update()

    def create_block(self, name, object_d):
        """ Initializes a ball.

        :param name: The name to refer to the object
        :type name: str
        :param object_d: Describes the objects appearance and location.
        :type object_d: dict
        """
        if object_d['shape'] == 'Ball':
            bpy.ops.mesh.primitive_ico_sphere_add(location=object_d['position'],
                                                  enter_editmode=False,
                                                  subdivisions=7,
                                                  radius = object_d['dims'][0])
        elif object_d['shape'] == 'Block':
            bpy.ops.mesh.primitive_cube_add(location=object_d['position'],
                                            enter_editmode=False,)
            ob = bpy.context.object
            self.scale_obj(ob, object_d['dims'])
            self.rotate_obj(ob, object_d['orientation'])
        else:
            raise ValueError('Not supported')


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

    def create_ramp(self, ramp_d):
        mesh = bpy.data.meshes.new('Ramp')  # add the new mesh
        obj = bpy.data.objects.new(mesh.name, mesh)
        col = bpy.data.collections.get("Collection 1")
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj

        x,y,z = ramp_d['dims']
        ratio = x / 2.0
        dz = np.sin(ramp_d['orientation'][1]) * x / ratio
        verts = [( 1.0,  1.0,  0.0),
                 ( 1.0, -1.0,  0.0),
                 (-1.0, -1.0,  0.0),
                 (-1.0,  1.0,  0.0),
                 ( 1.0,  1.0,  dz),
                 ( 1.0, -1.0,  dz),
                 (-1.0, -1.0,  dz),
                 (-1.0,  1.0,  dz)]
        edges = []
        faces = [[0, 1, 2, 3],
                [0, 4, 5, 1],
                [0, 4, 3],
                [1, 5, 2],
                [4, 5, 2, 3]]

        mesh.from_pydata(verts, [], faces)
        dx = np.cos(ramp_d['orientation'][1]) * x
        pos = [-dx/2, 0, 0]
        self.move_obj(obj, pos)
        self.rotate_obj(obj, [0, 0, np.pi])
        self.scale_obj(obj, [dx, y, dz*ratio])
        # self.set_appearance(obj, 'Ramp')

    def load_scene(self, scene_dict):
        """ Configures the ramp, table, and balls
        """
        # Setup ramp
        self.create_ramp(scene_dict['ramp'])
        # Setup table
        table = bpy.data.objects['TableGroup']
        table_d = scene_dict['table']
        self.move_obj(table, [0,0,-1*table_d['dims'][-1]])
        # self.rotate_obj(table, table_d['orientation'])
        # self.scale_obj(table, table_d['dims'])
        # Load Objects
        obj_names = list(scene_dict['objects'].keys())
        obj_names = sorted(obj_names)
        self.obj_names = obj_names
        for name, data in scene_dict['objects'].items():
            self.create_block(name, data)

    def set_rendering_params(self, resolution):
        """ Configures various settings for rendering such as resolution.
        """
        # bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.render.resolution_percentage = 100
        # bpy.context.scene.cycles.samples = 128
        # bpy.context.scene.render.tile_x = 16
        # bpy.context.scene.render.tile_y = 16

    def set_camera(self, rot):
        """ Moves the camera along a circular path.

        :param rot: Angle in radians along path.
        :type rot: float
        """
        # radius = 70.0
        # Move camera to position on ring
        # xyz = [np.cos(rot) * radius, np.sin(rot) * radius, 50]
        # camera = bpy.data.objects['Camera']
        # camera.location = xyz
        # bpy.context.view_layer.update()
        # The camera automatically tracks Empty
        # camera_track = bpy.data.objects['Empty']
        # self.move_obj(camera_track, [0, 0, 1])
        # camera.keyframe_insert(data_path='location', index = -1)
        # camera.keyframe_insert(data_path='rotation_quaternion', index = -1)


    def _frame_set(self,frame):
        """ Helper to `frame_set`.
        """
        positions = np.array(self.trace['pos'][frame])
        rotations = np.array(self.trace['orn'][frame])
        n_balls = len(positions)
        for ball_i in range(n_balls):
            obj_name = self.obj_names[ball_i]
            ball = bpy.data.objects[obj_name]
            self.move_obj(ball, positions[ball_i])
            self.rotate_obj(ball, rotations[ball_i])
            ball.keyframe_insert(data_path='location', index = -1)
            ball.keyframe_insert(data_path='rotation_quaternion', index = -1)


    def frame_set(self, frame, rot):
        """ Updates the scene to the given frame.

        :param frame: Index of keyframe
        :type frame: int
        :param rot: Rotation of camera
        :type rot: float
        """
        if frame < 0:
            frame = len(self.trace[0]['position']) + frame
        bpy.context.scene.frame_set(frame)
        n_sims = len(self.trace)
        print('Setting frame {0:d}'.format(frame))
        sys.stdout.flush()
        self._frame_set(frame)
        bpy.context.view_layer.update()
        print('Setting frame {0:d} ...done'.format(frame))
        sys.stdout.flush()


    def render(self, output_name, frames,
               resolution = (256, 256), camera_rot = None):
        """ Renders a scene.

        Skips over existing frames

        :param output_name: Path to save frames
        :type output_name: str
        :param frames: a list of frames to render (shifted by warmup)
        :type frames: list
        :param resolution: Image resolution
        :type resolution: tuple(int, int)
        :param camera_rot: Rotation for camera.
        :type camera_rot: float

        """
        if not os.path.isdir(output_name):
            os.mkdir(output_name)
        self.set_rendering_params(resolution)

        if camera_rot is None:
            camera_rot = np.zeros(len(frames))
        for i, (frame, cam) in enumerate(zip(frames, camera_rot)):
            out = os.path.join(output_name, '{0:d}'.format(i))
            if os.path.isfile(out + '.png'):
                msg = 'Frame {} already rendered at {}'
                msg = msg.format(i, out)
                print(msg)
                sys.stdout.flush()
                continue
            bpy.context.scene.render.filepath = out
            self.frame_set(frame, cam)
            t_0 = time.time()
            print('Rendering frame {0:d}'.format(i))
            sys.stdout.flush()
            with Suppressor():
                bpy.ops.render.render(write_still=True)
            dur = time.time() - t_0
            print('Rendering frame {} at {} took {}s'.format(i, out, dur))
            sys.stdout.flush()


    def save(self, out, frames):
        """
        Writes the scene as a blend file.
        """
        for i in range(frames):
            self.frame_set(i, self.theta)
        bpy.ops.wm.save_as_mainfile(filepath=out)

# From https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class Suppressor(object):

    # A context manager for doing a "deep suppression" of stdout and stderr in
    # Python, i.e. will suppress all print, even if the print originates in a
    # compiled C/Fortran sub-function.

    # This will not suppress raised exceptions, since exceptions are printed
    # to stderr just before a script exits, and after the context manager has
    # exited (at least, I think that is why it lets exceptions through).

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
    """Parses extra arguments
    """
    p = argparse.ArgumentParser(description = 'Renders blockworld scene')
    p.add_argument('--scene', type =load_data,
                   help = 'Tower json describing the scene.')
    p.add_argument('--trace', type = load_trace,
                   help = 'Trace json for physics.')
    p.add_argument('--out', type = str,
                   help = 'Path to save rendering')
    p.add_argument('--save_world', action = 'store_true',
                   help = 'Save the resulting blend scene')
    p.add_argument('--render_mode', type = str, default = 'default',
                   choices = ['default', 'none'],
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
    """Helper that loads trace file"""
    with open(path, 'r') as f:
        str = f.read()
        traces = json.loads(str)
    return traces

def load_data(path):
    """Helper that loads trace file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['scene']

def main():
    argv = sys.argv
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser(argv)

    scene = RampScene(args.scene, args.trace,
                      theta = args.theta)

    if args.gpu:
        print('Using gpu')
        bpy.context.scene.cycles.device = 'GPU'

    path = os.path.join(args.out, 'render')
    if not os.path.isdir(path):
        os.mkdir(path)

    if args.frames is None:
        n_frames = len(args.trace['pos'])
        frames = np.arange(n_frames)
    else:
        n_frames = len(args.frames)
        frames = args.frames

    if args.render_mode == 'default':
        scene.render(path, frames,
                     camera_rot = np.repeat(args.theta, n_frames),
                     resolution = args.resolution,)

    if args.save_world:
        path = os.path.join(args.out, 'world.blend')
        n_frames = len(args.trace['pos'])
        scene.save(path, n_frames)

if __name__ == '__main__':
    main()
