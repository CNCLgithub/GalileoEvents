import numpy as np
from copy import deepcopy
from itertools import repeat
from blockworld import towers, blocks
from blockworld.simulation import physics, generator, tower_scene
from blockworld.simulation.substances import Substance
from experiment.hypothesis.block_hypothesis import simulate

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_2vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
     12318300_0       3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class SimpleGen(generator.Generator):

    """
    Creates the configurations needed to evaluate critical blocks.
    """

    def __init__(self, materials, stability, block_size):
        self.materials = materials
        self.builder = stability
        self.block_size = np.array(block_size)

    def mutate_block(self, tower, subs, apps, idx, mat):
        """ Helper that allows  for indexed mutation.
        """
        mt = copy.deepcopy(subs)
        mt[idx] = Substance(mat).serialize()
        app = copy.deepcopy(apps)
        app[idx] = mat
        base = tower.apply_feature('appearance', app)
        return base.apply_feature('substance', mt)

    def configurations(self, tower, others = None):
        """
        Generator for different tower configurations.
        Arguments:
            tower (`dict`) : Serialized tower structure.
        Returns:
            A generator with the i-th iteration representing the i-th
            block in the tower being replaced.
            Each iteration contains a dictionary of tuples corresponding
            to a tower with the replaced block having congruent or incongruent
            substance to its appearance, organized with respect to congruent
            material.
            { 'mat_i' : [block_i,...,]
              ...
        """
        if others is None:
            others = self.unknowns
        subs = tower.extract_feature('substance')
        apps = tower.extract_feature('appearance')
        for block_i in range(len(tower)):
            d = {mat : self.mutate_block(tower, subs, apps, block_i, mat)
                 for mat in others}
            yield d

    def sample_blocks(self, n):
        """
        Procedurally generates blocks of cardinal orientations.
        """
        n = int(n)
        if n <= 0 :
            raise ValueError('n_blocks must be > 1.')
        for _ in range(n):
            block_dims = deepcopy(self.block_size)
            np.random.shuffle(block_dims)
            yield blocks.SimpleBlock(block_dims)

    def __call__(self, base, n_blocks):
        if not isinstance(base, towers.tower.Tower):
            try:
                base = list(base)
            except:
                raise ValueError('Unsupported base.')
            base = towers.EmptyTower(base)
        np.random.seed()
        return self.sample_tower(base, n_blocks)

class SimpleSim(physics.TowerEntropy):

    """
    Inherits `physic.TowerEntropy` to compute direction and
    stability information over a given tower.
    """

    def simulate(self, tower):
        """
        Controls simulations and extracts trace
        """
        return simulate(tower, self.frames)


    def direction(self, initial, final):
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
        vec = np.mean((final - initial)[:, :2], axis = 0)
        angle = angle_2vec((1,0), vec)
        mag = np.linalg.norm(vec)
        return angle, mag


    def stability(self, started, ended):
        """ Evaulates the stability of the tower.

        Defined in Battalgia 2013 as the proportion of
        blocks who have had a change in the z axis > 0.25
        """
        delta_z = abs(ended[:, 2] - started[:, 2])
        fall_ratio = np.sum(delta_z > 0.25) / len(delta_z)
        return fall_ratio

    def kinetic_energy(self, tower, positions):
        """ Computes KE
        """
        vel = physics.velocity(positions)
        vel = np.linalg.norm(vel, axis = -1)
        phys_params = tower.extract_feature('substance')
        density  = np.array([d['density'] for d in phys_params])
        volume = np.array([np.prod(tower.blocks[i+1]['block'].dimensions)
                           for i in range(len(tower))])
        mass = np.expand_dims(density * volume, axis = -1)
        # sum the vel^2 for each object across frames
        ke = 0.5 * np.dot(np.square(vel), mass).flatten()
        return np.sum(ke)

    def analyze(self, tower):
        """ Returns the stability statistics for a given tower.
        Args:
          tower : a `blockworld.towers.Tower`.
        Returns:
          A `dict` containing several statistics.
        """
        # original configuration
        trace = self.simulate(tower)
        pos = trace['position']
        angle, mag = self.direction(pos[0], pos[-1])
        stability = self.stability(pos[0], pos[-1])
        return {
            'angle' : angle,
            'mag' : mag,
            'instability' : stability,
            'trace' : trace,
        }

    def __call__(self, tower, configurations = None):
        """
        Evaluates the stability of the tower at each block.
        Returns:
          - The randomly sampled congruent tower.
          - The stability results for each block in the tower.
        """
        d = [{'id' : 'template', **self.analyze(tower)}]

        if not configurations is None:
            for b_id, conf in enumerate(configurations):
                mats = list(conf.keys())
                mat_towers = list(conf.values())
                kes = list(map(self.analyze, mat_towers))
                for idx, m in enumerate(mats):
                    id_str = '{0:d}_{1!s}'.format(b_id + 1, m)
                    d.append({'id' : id_str, **kes[idx]})
        return d
