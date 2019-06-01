""" Simulates ramp scenes in pybullet

Contains two main classes: Loader, RampPhysics

Loader - Interface for loading object data into scene
RampPhysics - Interface for simulating scenes
"""

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc

class Loader:

    """
    Interface for loading object data.
    """

    def make_table(params, p):
        mesh = p.GEOM_PLANE
        exs = np.array(params['dims']) / 2.0
        mass = 0
        friction = params['friction']
        base = p.createCollisionShape(mesh)
        wall_left = p.createCollisionShape(mesh,
                                           planeNormal = [0, 1, 0])
        wall_right = p.createCollisionShape(mesh,
                                            planeNormal = [0, 1, 0])
        wall_end = p.createCollisionShape(mesh,
                                          planeNormal = [1, 0, 0])
        positions = [(0, exs[1], 0), # left wall
                     (0, -1.0*exs[1]/2, 0), # right wall
                     (exs[0], 0, 0)] # end wall

        obj_id = p.createMultiBody(baseCollisionShapeIndex = base,
                                   linkPositions = positions,
                                   basePosition = pos,
                                   baseOrientation = rot)
        p.changeDynamics(obj_id, -1,
                         lateralFriction = friction)
        return obj_id

    def make_ramp(params, p):

        mesh = p.GEOM_PLANE
        exs = np.array(params['dims']) / 2.0
        mass = 0
        friction = params['friction']
        base = p.createCollisionShape(mesh)
        z_offset = np.sin(params['angle']) * exs[0]
        x_offset = np.cos(params['angle']) * exs[0] * -1.0
        pos = (x_offset, 0, z_offset)
        rot = p.getQuaternionFromEuler([0, params['angle'], 0])
        obj_id = p.createMultiBody(baseCollisionShapeIndex = base,
                                   basePosition = pos,
                                   baseOrientation = rot)
        p.changeDynamics(obj_id, -1,
                         lateralFriction = friction)
        return obj_id

    def make_ramp_obj(params, ramp, p):
        """
        A `pct` of 0 designates the `x` coordinate where the ramp touches
        the table.
        """
        if params['shape'] == 'block':
            mesh = p.GEOM_BOX
            dims = np.array(params['dims']) / 2.0
            z = dims[2]
            col_id = p.createCollisionShape(mesh,
                                            halfExtents = dims)
        else:
            mesh = p.GEOM_CYLINDER
            z = params['height'] / 2.0
            col_id = p.createCollisionShape(mesh,
                                            radius = params['radius'],
                                            height = params['height'])

        exs = np.array(ramp['dims'])
        mag = exs[0] * params['pct']
        z_offset = np.sin(ramp['angle']) * mag +\
            np.cos(ramp['angle']) * z
        x_offset = np.cos(ramp['angle']) * mag * -1.0 + \
            np.sin(ramp['angle']) * z

        pos = (x_offset, 0, z_offset)
        rot = p.getQuaternionFromEuler([0, ramp['angle'], 0])
        mass = params['mass']
        obj_id = p.createMultiBody(baseCollisionShapeIndex = col_id,
                                   basePosition = pos
                                   baseOrientation = rot)
        p.changeDynamics(obj_id, -1,
                         mass = mass,
                         lateralFriction = friction)
        return obj_id

    def make_table_obj(params, table, p):
        """
        A `pct` of 0 designates the `x` coordinate where the ramp touches
        the table.
        """
        if params['shape'] == 'block':
            mesh = p.GEOM_BOX
            dims = np.array(params['dims']) / 2.0
            col_id = p.createCollisionShape(mesh,
                                            halfExtents = dims)
        else:
            mesh = p.GEOM_CYLINDER
            col_id = p.createCollisionShape(mesh,
                                            radius = params['radius'],
                                            height = params['height'])

        mag = table['dims'][0] * params['pct']
        pos = (mag, 0, 0)
        mass = params['mass']
        obj_id = p.createMultiBody(baseCollisionShapeIndex = col_id,
                                   basePosition = pos)
        p.changeDynamics(obj_id, -1,
                         mass = mass,
                         lateralFriction = friction)
        return obj_id

class RampPhysics:

    """
    Handles physics for block towers.
    """

    def __init__(self, scene_json, loader = None):
        if loader is None:
            loader = Loader()
        self.loader = loader
        self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.world = tower_json

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
        d['ramp'] = self.loader.make_ramp_obj(w['ramp_obj'], w['ramp'])
        d['table'] = self.loader.make_table_obj(w['table_obj'], w['table'])
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
        (positions, rotations, ang_vel, lin_vel) = state
        for i, ob_id in enumerate(object_ids):
            p.resetBasePositionAndOrientation(ob_id,
                                              posObj = positions[i],
                                              ornObj = rotations[i])
            p.resetBaseVelocity(ob_id,
                                linearVelocity = lin_vel[i],
                                angularVelocity = ang_vel[i])

    def get_trace(self, frames, objects, time_step = 240, fps = 60,
                  state = None):
        """Obtains world state from simulation.

        Currently returns the position of each rigid body.
        The total duration is equal to `frames / fps`

        Arguments:
            frames (int): Number of frames to simulate.
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
            fixedTimeStep = 1.0 / time_step,
            enableConeFriction = 0)

        p.setGravity(0, 0, -10)

        positions = np.zeros((frames, len(objects), 3))
        rotations = np.zeros((frames, len(objects), 4))
        ang_vel = np.zeros((frames, len(objects), 3))
        lin_vel = np.zeros((frames, len(objects), 3))

        steps_per_frame = int(time_step / fps)
        total_steps = int(max(1, ((frames / fps) * time_step)))

        if not state is None:
            self.apply_state(object_ids, state)

        for step in range(total_steps):
            p.stepSimulation()

            if step % steps_per_frame != 0:
                continue

            for c, obj_id in enumerate(object_ids):
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                lin_vel, ang_vel = p.getBaseVelocity(obj_id)
                frame = np.floor(step / steps_per_frame).astype(int)
                positions[frame, c] = pos
                rotations[frame, c] = rot
                ang_vel[frame, c] = ang_vel
                lin_vel[frame, c] = lin_vel

        return (positions, rotations, ang_vel, lin_vel)
