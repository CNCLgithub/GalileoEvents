import itertools
import functools
import numpy as np

from .simple_gen import SimpleGen
from blockworld.simulation.substances import Substance

class MultiBlockGen(SimpleGen):

    """
    Inherits `SimpleGen` and extends mutations to include changing the
    mass of multiple blocks per tower.
    """

    def __init__(self, materials, stability, block_size, n = 1):
        self.materials = materials
        self.builder = stability
        self.block_size = np.array(block_size)
        self.n_blocks = n

    def mutate_block(self, tower, idx, mat):
        """ Helper that allows  for indexed mutation.
        """
        mt = tower.extract_feature('substance')
        mt[idx] = Substance(mat).serialize()
        app = tower.extract_feature('appearance')
        app[idx] = mat
        base = tower.apply_feature('appearance', app)
        return base.apply_feature('substance', mt)

    def configurations(self, tower, materials = None):
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
        if materials is None:
            materials = self.unknowns
        # Exclude all blocks on the first level
        _, levels = tower.levels()
        first_floor,_ = zip(*levels[0][1])
        block_ids = np.arange(len(tower))
        block_ids = np.setdiff1d(block_ids, list(first_floor))
        chunks = list(itertools.combinations(block_ids, self.n_blocks))
        chunks = np.array(chunks)
        np.random.shuffle(chunks)
        for blocks in chunks:
            d = {}
            for mat in materials:
                f = lambda t, bi: self.mutate_block(t, bi, mat)
                t = functools.reduce(f, blocks, tower)
                d[mat] = t
            yield (np.array(blocks) + 1, d)

