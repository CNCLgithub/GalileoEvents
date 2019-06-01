import copy

import dask
import numpy as np
from pprint import pprint

from compy import expression, graph
from mc.relation.hypothesis import Hypothesis
from blockworld.simulation import tower_scene
from blockworld import towers

from experiment.util import distribution

def digest_tower(template, *samples):
    """
    Applies samples from a random variable to a
    `towers.SimpleTower`.
    """
    t = template.serialize()
    for block in t:
        b_i = int(block['id'])
        if b_i == 0:
            continue
        features = samples[b_i - 1]
        for k,v in features.items():
            if k == 'density':
                block['data']['substance']['density'] = v[0]
            else: # position
                block['data']['pos'] = [v[k][0] for k in 'xyz']

    return towers.simple_tower.load(t)

def simulate(tower, frames = 120, fps = 30):
    """
    Controls simulations and extracts observations

    Arguments:
        tower (blockworld.Tower): A tower to run physics over
        frames (int, optional) : The number of frames to retrieve from physics
        fps (int, optional): The number of frames per second to capture.
    """
    sim = tower_scene.TowerPhysics(tower.serialize())
    trace = sim.get_trace(frames, tower.ordered_blocks, fps = fps)
    return trace

class Constant(expression.Expression):

    def __init__(self, val, name):
        self.val = val
        self._name = name

    @property
    def name(self):
        return self._name

    @dask.delayed
    def eval(self):
        return self.val

class BlockWorld(Hypothesis):

    """
    An instance of `Hypothesis` supporting the Block world
    experiment.

    Attributes:

      template (`dict`) : A source tower configuration.
      latents (`dict(expression.RandomVariable)`) : A `dict`
        enumerating which variables are latents.
      observations ([str]) : The list of observations to return.
      frames (int) : The range of frames to simulate.

    """

    def __init__(self, template, latents, observations, mutations,
                 sim_func = simulate):
        self.template = template
        self.latents = latents
        self.observations = observations
        self.mutations = mutations
        self.sim_func = sim_func


    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, t):
        if not isinstance(t, towers.Tower):
            raise ValueError('Template is not `Tower`.')
        self._template = t

    @property
    def latents(self):
        return self._latents

    @latents.setter
    def latents(self, d):

        block_keys = list(map(str, self.template.blocks.keys()))
        for rv in d:
            if rv.name not in block_keys:
                raise ValueError('Block {} not in template'.format(rv.name))
            if not isinstance(rv, expression.RandomVariable):
                raise ValueError('Block {} was not a RV'.format(rv.name))

        self._latents = d

    @property
    def observations(self):
        return self._obs

    @observations.setter
    def observations(self, o):
        self._obs = o

    def graph(self, frames):
        # wrap functions
        gen_tower = expression.Function(digest_tower)
        f_sim = expression.Function(self.sim_func)

        # declare variables
        t = Constant(frames, 'time')
        template = Constant(self.template, 'tower')
        # create a tower
        scene = expression.Application(gen_tower, template, *self.latents)
        # simulate
        obs = expression.Application(f_sim, scene, t)
        return graph.Graph(obs)


    def sample(self, frame):
        g = self.graph(frame + 1)
        # dask.visualize(g.source, filename = 'g')
        t = g.trace()
        obs_key = g.head
        obs = {o : t[obs_key][o] for o in self.observations}
        d = {
            'latents' : {rv.name : t[rv.name] for rv in self.latents},
            'observations' : obs,
        }
        return d

    def mutate(self, trace):
        """
        Returns a new `BlockWorld` belief with all mutable `latents`
        updated.

        Latents are updated according to the rules in `mutations`.
        For a given trace, each latent value is recast as a
        log-uniform distribution centered at the latents value
        from the trace.

        In addition, the `self.frames` is incremented by 1.
        """
        new_latents = []
        trace = trace['latents']

        for block_rv in self.latents:
            new_dist = copy.deepcopy(block_rv.dist)
            mutation = self.mutations[block_rv.name]
            # update the block's features.
            for feature in new_dist :
                if feature in mutation:
                    # apply particle shift
                    val = trace[block_rv.name][feature]
                    # print(block_rv.name, feature, val, mutation[feature])
                    new_val = mutation[feature](val)
                    new_dist[feature] = new_val

            new_latents.append(RandomBlock(block_rv.name, new_dist))

        return BlockWorld(self.template, new_latents, self.observations,
                          self.mutations)

    def pdf(self, trace):
        """Computes the pdf of the trace for each latent RV.
        """
        p = 1.0
        for block in self.latents:
            rv = self.latents[block]
            for feature in rv:
                if feature in self.mutations:
                    params = self.mutations[feature]
                    val = trace['latents'][block][feature]
                    p *= rv.dist[feature].prior(val)

        return p

class RandomBlock(expression.RandomVariable):

    """
    An instance of `RandomVariable` that represents
    a belief over the properties of a block in `towers.SimpleTower`.

    At the moment we are only considering mass (density since all volumes
    are the same).
    """

    feature_list = ['density', 'position']

    def __init__(self, name, params):
        self.name = name
        self.dist = params

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        self._name = s

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, d):

        if not all([f in self.feature_list for f in d]):
            raise ValueError('Some features not supported')

        for k in d:
            if not isinstance(d[k], distribution.Distribution):
                raise ValueError('{} must be a `Distribution`'.format(k))

        self._dist = d

    def sample(self):
        return {k : self.dist[k]() for k in self.dist}


