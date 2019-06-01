from .simple_gen import SimpleGen, SimpleSim
from ..simulation.noisy_scene import NoisyPhysics

import numpy as np
import networkx as nx
from blockworld.simulation.tower_scene import Loader

class ShiftedSim(SimpleSim):

    """
    Inherits `SimpleSim` to compute direction and
    stability information over a given tower and random
    perturbation to xy coordinates for each block.
    """

    def direction(self, trace):
        """ Determines the direction that the tower falls.

        Defined in Battalgia 2013 as the average final
        positions of all blocks in the xy plane.

        Adapted from David Wolever @ https://bit.ly/2rQe1Fu

        Arguments:
           final (np.ndarray): Final positions of each block.

        Returns:
           (angle, mag) : A tuple containing the angle of falling
        and its magnitude.
        """
        pos = trace['position']
        return super().direction(pos[0], pos[-1])


    def stability(self, trace):
        """ Evaulates the stability of the tower.

        Defined in Battalgia 2013 as the proportion of
        blocks who have had a change in the z axis > 0.25
        """
        pos = trace['position']
        return super().stability(pos[0], pos[-1])

    def analyze(self, tower):
        """ Returns the stability statistics for a given tower.
        Args:
          tower : a `blockworld.towers.Tower`.
        Returns:
          A `dict` containing several statistics.
        """
        # original configuration
        trace = self.simulate(tower)
        angle, mag = self.direction(trace)
        stats = {
            'angle' : angle,
            'mag' : mag,
            'instability' : self.stability(trace),
        }
        # Stats from noisy xy shifts
        perturbations = self.perturb(tower, n = 10)
        traces = list(map(self.simulate, perturbations))
        noisy_instability = list(map(self.stability, traces))
        stats['instability_p'] = noisy_instability
        stats['instability_mu'] = np.mean(noisy_instability)
        return (trace, stats)

class PushLoader(Loader):

    """
    Interface for injecting random forces to initial frame.

    Applies the force vector generated from the given function `force_func`
    (can be deterministic), to each block in the scene. 
    """
    def __init__(self, force_vec):
        self.force_vec = force_vec

    def __call__(self, name, start, p):

        rot = p.getQuaternionFromEuler([0, 0, 0])
        if name == 0:
            mesh = p.GEOM_PLANE
            radius = 10
            col_id = p.createCollisionShape(mesh, radius = radius)
            pos = [0., 0., 0.]
            mass = 0
            friction = 0.5
            force = [0., 0., 0.,]
        else:
            mesh = p.GEOM_BOX
            dims = np.array(start['dims']) / 2.0
            col_id = p.createCollisionShape(mesh,
                                            halfExtents = dims,
                                            )
            pos = start['pos']
            mass = np.prod(start['dims']) * start['substance']['density']
            friction = start['substance']['friction']
            force = self.force_vec

        obj_id = p.createMultiBody(mass, col_id, -1, pos, rot)
        p.changeDynamics(obj_id, -1, lateralFriction = friction)
        # Applies for to center of object
        p.applyExternalForce(obj_id, -1, force, [0., 0., 0.], p.WORLD_FRAME)
        return obj_id

def get_bottom_blocks(tower):
    """
    Helper that returns the ids of blocks touching the ground
    """
    g = nx.subgraph(tower.graph.copy(), np.arange(1, len(tower) + 1))
    ids = []
    for i in tower.ordered_blocks:
        if len(g.pred[i]) == 0:
            ids.append(i)
    return ids


class PushedSim(ShiftedSim):

    """
    Inherits `SimpleSim` to compute direction and
    stability information over a given tower with a random
    force vector applied to each block at the initial frame.
    """

    def simulate(self, tower, to_push = None, force = None,
                 window = 0, frames = 240, fps = 30):
        """
        Controls simulations and extracts trace
        Arguments:
        tower (blockworld.Tower): A tower to run physics over
        frames (int, optional) : The number of frames to retrieve from physics
        fps (int, optional): The number of frames per second to capture.
        """
        sim = NoisyPhysics(tower.serialize(), force = force)
        trace = sim.get_trace(frames, tower.ordered_blocks,
                              push_blocks = to_push,
                              push_window = window, fps = fps)
        return trace

    def sample_force(self, force):
        """
        Samples a force vector in the xy plane
        """
        force_theta = np.random.uniform(0, np.pi*2)
        f_x = force * np.cos(force_theta)
        f_y = force * np.sin(force_theta)
        return [f_x, f_y, 0.]


    def analyze(self, tower, k = 10, force = 1.0,
                window = 10):
        """ Returns the stability statistics for a given tower.
        Args:
          tower : a `blockworld.towers.Tower`.
          k     : Number of simulations to run.
        Returns:
          A `dict` containing several statistics.
        """
        # blocks to push in noisy condition
        to_push = get_bottom_blocks(tower)
        # original configuration
        trace = self.simulate(tower)
        angle, mag = self.direction(trace)
        stats = {
            'angle' : angle,
            'mag' : mag,
            'instability' : self.stability(trace),
        }
        # Stats from random pushed
        traces = np.empty((k,), dtype = object)
        perturbations = self.perturb(tower, n = k)
        for i in range(k):
            force_vec = self.sample_force(force)
            traces[i] = self.simulate(perturbations[i],
                                      force = force_vec,
                                      to_push = to_push,
                                      window = window)

        noisy_instability = list(map(self.stability, traces))
        stats['instability_p'] = noisy_instability
        stats['instability_mu'] = np.mean(noisy_instability)
        return (trace, stats, traces)

def main():
    # generate a random tower
    import json
    from pprint import pprint
    from .multi_gen import MultiBlockGen
    from experiment.render import interface
    materials = {'Wood' : 1.0}
    gen = MultiBlockGen(materials, 'local', (3,1,1), 2)
    force = 1000.0
    noise = 0.15
    phys = PushedSim(noise = noise, frames = 240)
    tower = gen((1,1), 10)
    # simulate
    trace, stats, traces = phys.analyze(tower, force = force, k = 30)
    while stats['instability'] > 0:
        tower = gen((1,1), 10)
        trace, stats, traces = phys.analyze(tower, force = force, k = 30)
    mat_path = 'experiment/render/materials.blend'
    scene_str = json.dumps(tower.serialize())
    # interface.render(scene_str, [trace], 0.0, 'original', 'none',
    #                  None, mat_path)
    interface.render(scene_str, traces, 0.0, 'pushed', 'none',
                     None, mat_path)
    pprint(stats)
if __name__ == '__main__':
    main()
