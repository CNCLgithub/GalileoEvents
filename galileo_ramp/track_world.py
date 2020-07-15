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

def pct_to_coord(mag, angle, z):
    y_offset = (np.cos(angle) * mag) + (np.sin(angle) * z)
    z_offset = (np.sin(angle) * abs(mag)) + (np.cos(angle) * z)
    return np.array([0, y_offset, z_offset])

def pct_to_coord_track(mag, z):
    diameter = 0.903725
    # circle constnats
    radius = diameter / 2 
    center = np.array([0, radius, 0])
    # gets angle based on magnitude given
    angle = mag * 2 * np.pi
    # calculates x, y coordinates based on angle
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    return np.array([x_coord, y_coord, z]) + center

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
                 ramp_angle = 43.0 * (np.pi/180.),
                 ramp_phys = default_phys,
                 track_source = track_path):

        ramp = Block('ramp', ramp_dims, ramp_phys, angle = (0, ramp_angle, 0))
        ramp.position = [0, -0.34536, 0.27761]
        self.ramp = ramp
        self.ramp_angle = ramp_angle
        track = MeshShape(track_source, {'lateralFriction': 0.1, 'density': 0.0,
                                         'rollingFriction':1.0,
                                         'restitution' : 1.0})
        track.position = [0., 0., 0.]
        self.objects = None
        self.track = track
        self.initial_pos = {}
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
        if place < 1:
            angle = 0
            pos = pct_to_coord_track(place, z/2) 
            print(pos)
        # on ramp
        elif place < 2 and place > 1:
            angle = self.ramp_angle
            mag = (1 - place) * self.ramp.dimensions[0]
            pos = pct_to_coord(mag, angle, z/2) + self.ramp.position
        else:
            raise ValueError('Place not found')

        obj.position = pos
        obj.orientation = (0, angle, 0)
        objects = self.objects
        objects[name] = obj
        self.objects = objects
        initial_pos = self.initial_pos
        initial_pos[name] = place
        self.initial_pos = initial_pos
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
