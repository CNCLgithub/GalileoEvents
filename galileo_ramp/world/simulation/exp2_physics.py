""" Simulates ramp scenes in pybullet

Contains two main classes: Loader, RampPhysics

Loader - Interface for loading object data into scene
RampPhysics - Interface for simulating scenes
"""

import time
import numpy as np
import operator as op
from functools import reduce
from itertools import combinations

import pybullet
import pybullet_utils.bullet_client as bc

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

class Loader:

    """
    Interface for loading object data.
    """

    def make_table(self, params, p):
        mesh = p.GEOM_BOX
        exs = np.array(params['dims']) / 2.0
        mass = 0
        rot = p.getQuaternionFromEuler(params['orientation'])
        friction = params['friction']
        base = p.createCollisionShape(mesh, halfExtents = exs)
        base_id = p.createMultiBody(baseCollisionShapeIndex = base,
                                    basePosition = params['position'],
                                    baseOrientation = rot,)
        p.changeDynamics(base_id, -1, lateralFriction = friction)

        rot = p.getQuaternionFromEuler((np.pi/2, 0, 0))
        wall_left = p.createCollisionShape(mesh, halfExtents = exs)
        obj_id = p.createMultiBody(baseCollisionShapeIndex = wall_left,
                                   basePosition = [params['position'][0], exs[1], 0],
                                   baseOrientation = rot,)
        p.changeDynamics(obj_id, -1)
        wall_right = p.createCollisionShape(mesh, halfExtents = exs)
        obj_id = p.createMultiBody(baseCollisionShapeIndex = wall_right,
                                   basePosition = [params['position'][0], -1 * exs[1], 0],
                                   baseOrientation = rot,)
        p.changeDynamics(obj_id, -1)
        rot = p.getQuaternionFromEuler((0, np.pi/2, 0))
        wall_end = p.createCollisionShape(mesh, halfExtents = exs)
        obj_id = p.createMultiBody(baseCollisionShapeIndex = wall_end,
                                   basePosition = [exs[0]*2+exs[2], 0, 0],
                                   baseOrientation = rot,)
        p.changeDynamics(obj_id, -1)
        return base_id

    def make_ramp(self, params, p):

        mesh = p.GEOM_BOX
        mass = 0
        friction = params['friction']
        exs = np.array(params['dims'])/2.0
        base = p.createCollisionShape(mesh, halfExtents = exs)
        rot = p.getQuaternionFromEuler(params['orientation'])
        obj_id = p.createMultiBody(baseCollisionShapeIndex = base,
                                   basePosition = params['position'],
                                   baseOrientation = rot)
        p.changeDynamics(obj_id, -1,
                         lateralFriction = friction)
        return obj_id

    def make_obj(self, params, p):
        if params['shape'] == 'Block':
            mesh = p.GEOM_BOX
            dims = np.array(params['dims']) / 2.0
            col_id = p.createCollisionShape(mesh, halfExtents = dims)
        elif params['shape'] == 'Ball':
            mesh = p.GEOM_SPHERE
            z = params['dims'][0]
            col_id = p.createCollisionShape(mesh, radius = z)
        else:
            mesh = p.GEOM_CYLINDER
            z = params['height'] / 2.0
            col_id = p.createCollisionShape(mesh,
                                            radius = params['radius'],
                                            height = params['height'])

        rot = p.getQuaternionFromEuler(params['orientation'])
        obj_id = p.createMultiBody(baseCollisionShapeIndex = col_id,
                                   basePosition = params['position'],
                                   baseOrientation = rot)
        p.changeDynamics(obj_id, -1,
                         mass = params['mass'],
                         lateralFriction = params['friction'],
                         rollingFriction = 0.1,
                         linearDamping =  0.006,
                         restitution = 0.50)
        return obj_id


class RampPhysics:

    """
    Handles physics for block towers.
    """

    def __init__(self, scene_json, loader = None, debug = False):
        if loader is None:
            loader = Loader()
        self.loader = loader
        if debug:
            self.client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.debug = debug
        self.world = scene_json
        if 'gravity' in scene_json:
            self.client.setGravity(0, 0, scene_json['gravity'])
        else:
            self.client.setGravity(0, 0, -10)

    #-------------------------------------------------------------------------#
    # Attributes

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, l):
        if not isinstance(l, Loader):
            raise TypeError('Loader has wrong type')
        self._loader = l


    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, w):
        self.client.resetSimulation()
        self.loader.make_ramp(w['ramp'], self.client)
        self.loader.make_table(w['table'], self.client)
        d = {}
        for obj,data in w['objects'].items():
            d[obj] = self.loader.make_obj(data, self.client)
        self._world = d

    #-------------------------------------------------------------------------#
    # Methods

    def apply_state(self, object_ids, state):
        """ Applies a state matrix to each object reported

        The expected dimensions of state are obsxSTATE
        where `obs` is the number of objects (in order)
        and STATE is the tuple representing the dimensions of the
        component.

        :param object_ids: A list of ids corresponding objects in `state`
        :param state: A tuple of position, rotation, angular velocity
        and linear velocity for each object.
        """
        p = self.client
        (positions, rotations, ang_vel, lin_vel, _) = state
        for i, ob_id in enumerate(object_ids):
            p.resetBasePositionAndOrientation(ob_id,
                                              posObj = positions[i],
                                              ornObj = rotations[i])
            p.resetBaseVelocity(ob_id,
                                linearVelocity = lin_vel[i],
                                angularVelocity = ang_vel[i])

    def get_trace(self, dur, objects, time_step = 240, fps = 60,
                  time_scale = 1.0, state = None):
        """Obtains world state from simulation.

        Currently returns the position of each rigid body.
        The total duration is equal to `frames / fps`

        Arguments:
            dur (float): The amount of time in seconds to simulate
            objects ([str]): List of strings of objects to report
            time_step (int, optional): Number of physics steps per second
            fps (int, optional): Number of frames to report per second.
        """
        for obj in objects:
            if not obj in self.world.keys():
                raise ValueError('Object {} not found'.format(obj))
        object_ids = [self.world[obj] for obj in objects]

        p = self.client
        p.setPhysicsEngineParameter(
            enableFileCaching = 0,
            fixedTimeStep = 1/time_step
        )

        # dur = dur / time_scale
        # fps = int(np.ceil(fps / time_scale))
        dt = 1.0 / fps * time_scale
        frames = int(np.floor(dur * fps))

        positions = np.zeros((frames, len(objects), 3))
        rotations = np.zeros((frames, len(objects), 4))
        ang_vel = np.zeros((frames, len(objects), 3))
        lin_vel = np.zeros((frames, len(objects), 3))
        collisions = np.zeros((frames, ncr(len(objects), 2)))

        if not state is None:
            self.apply_state(object_ids, state)

        if self.debug:
            p.setRealTimeSimulation(1)

            while (1):
                keys = p.getKeyboardEvents()
                print(keys)

                time.sleep(0.01)
            return

        for frame in range(frames):
            self.step_simulation(dt, time_step)

            for c,(a,b) in enumerate(combinations(object_ids, 2)):
                cp = len(p.getContactPoints(bodyA=a, bodyB=b))
                collisions[frame, c] += cp

            for c, obj_id in enumerate(object_ids):
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                l_vel, a_vel = p.getBaseVelocity(obj_id)
                positions[frame, c] = pos
                rotations[frame, c] = rot
                ang_vel[frame, c] = a_vel
                lin_vel[frame, c] = l_vel

        return (positions, rotations, ang_vel, lin_vel, collisions)


    def step_simulation(self, dt, time_step):
        current = 0.0
        while current < dt:
            self.client.stepSimulation()
            current += 1.0/time_step



def run_full_trace(data, objects, T = 1.0, fps = 60, time_scale = 1.0):
    t = RampPhysics(data)
    return t.get_trace(T, objects, fps = fps, time_scale = time_scale)

def run_mc_trace(data, objects, state = None, pad = 0, fps = 6):
    t = RampPhysics(data)
    steps = 2*pad + 1
    current_trace = t.get_trace(steps, objects,  fps = fps, state = state)
    smooth_trace = lambda x: x[0]
    trace = list(map(smooth_trace, current_trace))
    return trace
