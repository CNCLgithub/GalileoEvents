import numpy as np
from collections import OrderedDict
from rbw.worlds import World
from rbw.shapes import Shape, Block

default_phys = {
    'lateralFriction': 0.2,
    'density': 0.0,
    'restitution': 0.9
        }

track_path = ""

class MeshShape(Shape):

    def __init__(self, source, physics):
        super().__init__("", [0.,0.,0.,], physics)
        self.source = source

    @property
    def shape(self):
        """ The name of the shape
        :returns: A shape name
        :rtype: str
        """
        return "Mesh"

    @property
    def volume(self):
        """
        Mesh volume returns 0
        :returns: the volume the shape holds
        :rtype: float
        """
        return 0

    @property
    def dimensions(self):
        return [0., 0., 0.]
    @dimensions.setter
    def dimensions(self, value):
        pass

    def serialize(self):
        """ Returns a json friendly dictionary representation
        :rtype: dict
        """
        d = super().serialize()
        d['source'] = self.source
        return d

class TrackWorld(World):
    """
    Describes a track scene
    """

    def __init__(self, outer_track_radius, inner_track_radius, ramp_dims,
                 ramp_angle = 43.0,
                 ramp_phys = default_phys,
                 track_source = track_path):

        ramp = Block('ramp', ramp_dims, ramp_phys, angle = (0, ramp_angle, 0))
        ramp.position = [0, -4.2582, 3.2272]
        self.ramp = ramp
        track = MeshShape(track_source, {'lateralFriction': 0.0, 'density': 0.0,
                                         'restitution' : 0.9})
        track.position = [0., 0., 0.]
        self.objects = None
        self.track = track
        self.init_vel = {}
        self.init_force = {}

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, v):
        if v is None:
            v = OrderedDict()
        self._objects = v
   
    def add_object(self, name, obj, place,
                   vel = None, force = None):
        z = obj.dimensions[-1]
        # on track
        # on ramp
        if not vel is None:
            self.init_vel[name] = vel
        if not force is None:
            self.init_force[name] = force

    def serialize(self):
        """Returns a json-serialized context of the world"""
        d = super().serialize()
        d['track'] = self.track.serialize()
        d['init_vel'] = self.init_vel
        d['init_force'] = self.init_force
        return d
