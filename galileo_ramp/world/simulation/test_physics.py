""" A minimal physical scene used for validation

Contains one class: TestPhysics
"""
import copy
import time
import numpy as np
import operator as op

import pybullet
import pybullet_utils.bullet_client as bc

table_params = {
    'dims' : [10, 10, 1],
    'position': [5, 0, -0.5],
    'orientation': [0, 0, 0],
    'friction': 0.5
}
ramp_params = {
    'dims' : [10, 10, 1],
    'position': [-5, 0, 5],
    'orientation': [0, np.pi/4, 0],
    'friction': 0.5
}

def make_table(params, p):
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
    return base_id

def make_ramp(params, p):

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

def make_obj(params, p):
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
                     rollingFriction = 0.05,
                     restitution = 1.0)
    return obj_id

obj_params_t = {
    'dims' : [1, 1, 1],
    'position': [-5, 0, 6.5],
    'orientation': [0, np.pi/4, 0],
    'friction': 0.5,
    'shape': 'Block',
    'mass': 1.0
        }
class TestPhysics:

    def __init__(self, obj_data = None, debug = False):
        if debug:
            self.client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.debug = debug
        self.table_id = make_table(table_params, self.client)
        self.ramp_id = make_ramp(ramp_params, self.client)
        obj_params = copy.deepcopy(obj_params_t)
        if obj_data is None:
            obj_data = {}
        for k in obj_data:
            obj_params[k] = obj_data[k]
        self.obj_id = make_obj(obj_params, self.client)

    #-------------------------------------------------------------------------#
    # Methods

    def apply_state(self, state):
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
        p.resetBasePositionAndOrientation(self.obj_id,
                                          posObj = np.mean(positions, axis = 0),
                                          ornObj = np.mean(rotations, axis = 0))
        p.resetBaseVelocity(self.obj_id,
                            linearVelocity = np.mean(lin_vel, axis = 0),
                            angularVelocity = np.mean(ang_vel, axis = 0))

    def get_trace(self, frames, time_step = 240, fps = 60,
                  state = None):
        """Obtains world state from simulation.

        Currently returns the position of each rigid body.
        The total duration is equal to `frames / fps`

        Arguments:
            frames (int): Number of frames to simulate.
            time_step (int, optional): Number of physics steps per second
            fps (int, optional): Number of frames to report per second.
        """

        p = self.client
        p.setPhysicsEngineParameter(
            enableFileCaching = 0,
        )

        p.setGravity(0, 0, -10)


        steps_per_frame = int(time_step / fps)
        total_steps = int(max(steps_per_frame, steps_per_frame*frames ))

        positions = np.zeros((frames, 3))
        rotations = np.zeros((frames, 4))
        ang_vel = np.zeros((frames, 3))
        lin_vel = np.zeros((frames, 3))

        pad_positions = np.zeros((steps_per_frame, 3))
        pad_rotations = np.zeros((steps_per_frame, 4))
        pad_ang_vel = np.zeros((steps_per_frame, 3))
        pad_lin_vel = np.zeros((steps_per_frame, 3))


        if not state is None:
            self.apply_state(state)

        if self.debug:
            p.setRealTimeSimulation(1)
            while (1):
                keys = p.getKeyboardEvents()
                print(keys)
                time.sleep(0.01)
            return

        # print(total_steps, steps_per_frame)
        for step in range(total_steps):
            p.stepSimulation()

            pos, rot = p.getBasePositionAndOrientation(self.obj_id)
            l_vel, a_vel = p.getBaseVelocity(self.obj_id)

            pad_positions[step % steps_per_frame] = pos
            pad_rotations[step % steps_per_frame] = rot
            pad_ang_vel[step % steps_per_frame] = a_vel
            pad_lin_vel[step % steps_per_frame] = l_vel

            record = (step+1) % steps_per_frame == 0
            if record:
                frame = np.floor(step / steps_per_frame).astype(int)
                # positions[frame] = np.mean(pad_positions, axis = 0)
                # rotations[frame] = np.mean(pad_rotations, axis = 0)
                positions[frame] = pad_positions[-1]
                rotations[frame] = pad_rotations[-1]
                ang_vel[frame] = pad_ang_vel[-1]
                lin_vel[frame] = pad_lin_vel[-1]
                # ang_vel[frame] = np.mean(pad_ang_vel, axis = 0)
                # lin_vel[frame] = np.mean(pad_lin_vel, axis = 0)

        return (positions, rotations, ang_vel, lin_vel)

def run_full_trace(T, data):
    t = TestPhysics(obj_data = data)
    return t.get_trace(T, fps = 6)

def run_mc_trace(state = None, T = 1, pad = 1, fps = 6,
                 data = None):
    t = TestPhysics(obj_data = data)
    traces = []
    steps = 2*pad + 1
    current_trace = t.get_trace(steps, fps = fps, state = state)
    smooth_trace = lambda x: x[0]
    traces.append(list(map(smooth_trace, current_trace)))
    for _ in range(T-1):
        current_trace = t.get_trace(steps, state = current_trace, fps = fps)
        traces.append(list(map(smooth_trace, current_trace)))

    return list(map(np.vstack, zip(*traces)))
