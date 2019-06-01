from __future__ import division
import math
import warnings
import scipy.stats
import numpy as np
from abc import ABC, abstractmethod

class Distribution(ABC):

    '''
    Abstract interface for distributions
        methods:
            - __call__ : Samples from distribution with parameters `psi`
            - drift : Returns a new distribution with drift added to `psi`
    '''

    @abstractmethod
    def __call__(self, n):
        pass

    @abstractmethod
    def drift(self):
        '''
        Adds drift to distribution parameters
        '''
        pass

    @abstractmethod
    def serialize(self):
        '''
        Returns a dictionary represenation
        '''
        pass

    @abstractmethod
    def prior(self, v):
        '''
        Returns the probability that a given value came from the distribution
        '''
        pass

class Lambda(Distribution):
    def __init__(self, func, dist):
        self.func = func
        self.dist = dist

    def __call__(self, n = 1):
        samples = np.array(list(map(self.func, self.dist(n=n))))
        return samples

    def prior(self, v):
        warnings.warn('Computing the prior over a `LambdaDist` may result' +\
                      'in incorrect behavior')
        return self.dist.prior(v)

    def drift(self):
        pass

    def serialize(self):
        base = self.dist.serialize()
        base['func'] = self.func.__name__
        return base

class Uniform(Distribution):
    '''
    Implementation of a 1-D uniform distribution
        attributes:
            - lo : The lower bound of the space
            - hi : The upper bound
            - noise : The change of the mean of the distribution
    '''

    def __init__(self, lo, hi, noise = 0.0, steps = 50):

        self.lo = min(lo, hi)
        self.hi = max(lo, hi)
        self.steps = steps
        self.noise = noise

    #---------------------------------------------------------------------#
    def __call__(self, n = 1):
        rng = np.linspace(self.lo, self.hi, num=self.steps)
        samples = np.random.choice(rng, size = n)
        return samples

    def prior(self, v):
        if v >= self.lo and v < self.hi:
            return 1. / (self.hi - self.lo)
        else:
            return 0.
    '''
    Uses a previously sampled value to create a new distribution with the same
    width.
    '''

    def drift(self, new_mu):
        d = abs(self.hi - self.lo) / 2.0
        new_lo = new_mu - d
        new_hi = new_mu + d
        return UniformDist(new_lo, new_hi, noise = self.noise)


    def serialize(self):
        return {"lo" : self.lo, "hi" : self.hi, "noise" : self.noise,
            "type" : self.__class__.__name__}

class Constant(Distribution):
    '''
    A template implementation of `Distribution` for parameters that will not
    change with sampling or drift.
    '''
    def __init__(self, value):
        self.value = value

    def __call__(self, n = 1):
        samples = np.repeat(self.value, n)
        return samples

    def prior(self, v):
        if v == self.value:
            return 1.
        else:
            return 0.

    def drift(self, v):
        return ConstantDist(self.value)

    def serialize(self):
        return {"type" : "constant"}

class Normal(Distribution):
    '''
    Normal distribution centered around `mu` with std of `sigma`
    '''
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, n = 1):
        samples = np.random.normal(self.mu, self.sigma, n)
        return samples

    def prior(self, x):
        mu = self.mu
        sigma = self.sigma
        X = scipy.stats.norm(loc = mu, scale=sigma)
        return X.pdf(x)

    def drift(self, v):
        return NormalDist(v, self.sigma)

    def serialize(self):
        return {"type" : self.__class__.__name__, "mu" : self.mu,
            "std" : self.sigma}

class TruncNorm(Distribution):
    '''
    Truncated Gaussian distribution
    '''
    def __init__(self, mu, sigma, a, b):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

    def __call__(self, n = 1):
        lo = (self.a - self.mu) / self.sigma
        hi = (self.b - self.mu) / self.sigma
        X = scipy.stats.truncnorm(lo, hi, loc= self.mu, scale=self.sigma)
        samples = X.rvs(n)
        return samples

    def prior(self, x):
        # truncnorm only ranges > 0
        lo = (self.a - self.mu) / self.sigma
        hi = (self.b - self.mu) / self.sigma
        nx = (x - self.mu) / self.sigma
        X = scipy.stats.truncnorm(lo, hi, loc= self.mu, scale=self.sigma)
        pdf = X.pdf(x)
        return pdf

    def drift(self, v):
        return TruncGauss(v, self.sigma, self.a, self.b)

    def serialize(self):
        return {"type" : self.__class__.__name__, "mu" : self.mu,
            "std" : self.sigma, "a":self.a, "b":self.b}

class MixtureDist(Distribution):
    '''
    A multinomial mixture model that contains members of `Distribution`.
    The pdf is not a true pdf in the sense that every component is summed
    directly rather than summing to 1.
    '''
    '''
        input: pairs -> A list of tuples containing a member of `Distribution`
            and its corresponding weight
    '''
    def __init__(self, dists, weights = None):


        self.dists = dists
        self.weights = weights

    def __call__(self, n = 1):
        if not self.weights is None:
            choices = np.random.choice(self.dists, n, p = self.weights)
        else:
            choices = np.random.choice(self.dists, n)

        return [d(1) for d in choices]

    def prior(self, v):
        p = 0.
        # if no weights are provide, assume equality
        if self.weights is None:
            weights = np.repeat(1.0 / len(self.dists) , len(self.dists))
        else:
            weights = self.weights
        # compute the scaled pdf of each component and sum them.
        for dist, w in zip(self.dists, weights):
            p_d = dist.prior(v)
            p = np.sum([p, w * p_d])

        return p

    def drift(self, v):
        new_ds = [d.drift(v) for d in self.dists]
        return MixtureDist(new_ds, self.weights)

    def serialize(self):
        base  = {"type" : self.__class__.__name__, "weights" : self.weights}
        for i,d in enumerate(self.dists):
            base.update({'{0:d}'.format(i) : d.serialize()})
        return base


class Collection(Distribution):
    '''
    A collection of `Distribution`s.
    '''
    def __init__(self, ds):
        if not isinstance(ds, dict):
            msg = 'Collection must be initialized with a dict of `Distributions`'
            raise ValueError(msg)
        if not all(map(lambda x: isinstance(x, Distribution), ds.values())):
            msg = 'All values must be of type `Distribution`'
            raise ValueError(msg)
        self.ds = ds


    def __call__(self, n = 1):
        return {k : self.ds[k](n=n) for k in self.ds}

    def prior(self, xs):
        if not isinstance(xs, dict):
            raise ValueError('prior must receive `dict`')
        return {k : self.ds[k].prior(xs[k]) for k in self.ds}

    def drift(self, vs):
        if not isinstance(xs, dict):
            raise ValueError('drift must receive `dict`')
        return {k : self.ds[k].drift(vs[k]) for k in self.ds}

    def serialize(self):
        return {
            'type' : self.__class__.__name__,
            'ds' : {k : self.ds[k].serialize() for k in self.ds}
        }
