import numpy as np
from blockworld.simulation import tower_scene

class NoisyPhysics(tower_scene.TowerPhysics):

    """
    Tower scene with added force to the bottom blocks.
    """

    def __init__(self, *args, force = 0.0, **kwargs):
        self.force = force
        super(NoisyPhysics, self).__init__(*args, **kwargs)

    def get_trace(self, frames, objects, time_step = 240, fps = 60,
                  push_blocks = None, push_window = 0):
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
                raise ValueError('Block {} not found'.format(obj))
        object_ids = [self.world[obj] for obj in objects]
        if push_blocks is None:
            push_blocks = []
        push_blocks = [self.world[o] for o in push_blocks]

        p = self.client
        p.setPhysicsEngineParameter(
            fixedTimeStep = 1.0 / time_step,
            enableConeFriction = 0)
        p.setGravity(0, 0, -10)

        positions = np.zeros((frames, len(objects), 3))
        rotations = np.zeros((frames, len(objects), 4))

        steps_per_frame = int(time_step / fps)
        total_steps = int(max(1, ((frames / fps) * time_step)))
        for step in range(total_steps):

            if step < push_window:
                # Apply forces
                for obj_id in push_blocks:
                    p.applyExternalForce(obj_id, -1, self.force,
                                         [0., 0., 0.], p.WORLD_FRAME)

            p.stepSimulation()

            if step % steps_per_frame != 0:
                continue

            for c, obj_id in enumerate(object_ids):
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                frame = np.floor(step / steps_per_frame).astype(int)
                positions[frame, c] = pos
                rotations[frame, c] = rot

        result = {'position' : positions, 'rotation' : rotations}
        return result
