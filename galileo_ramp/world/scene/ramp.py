from collections import OrderedDict
import numpy as np

from . import block


def pct_to_coord(mag, angle, z):
    x_offset = (np.cos(angle) * mag) + (np.sin(angle) * z)
    z_offset = (np.sin(angle) * abs(mag)) + (np.cos(angle) * z)
    return np.array([x_offset, 0, z_offset])

class RampScene:

    """
    Describes a ramp scene
    """

    def __init__(self, table_dims, ramp_dims, objects = None,
                 ramp_angle = 0.0, table_friction = 0.8,
                 ramp_friction = 0.8):
        table = block.Block('table', (*table_dims, 1), 0, table_friction)
        # We want the table surface to be the xy plane @ z = 0
        table.position = pct_to_coord(table_dims[0]*0.5 , 0, 0)  - np.array((0, 0, 0.5))
        self.table = table
        ramp = block.Block('ramp', (*ramp_dims, 1), 0, ramp_friction,
                                angle = (0, ramp_angle, 0))
        ramp.position = pct_to_coord(ramp_dims[0]*(-0.5), ramp_angle, 0.0) - np.array((0, 0, 0.5))
        self.ramp = ramp
        self.ramp_angle = ramp_angle
        self.objects = objects

    @property
    def objects(self):
        if self._objects is None:
            return OrderedDict()
        else:
            return self._objects

    @objects.setter
    def objects(self, v):
        self._objects = v

    def add_object(self, name, obj, place):
        z = obj.dimensions[-1]
        # on table
        if place < 1:
            mag = place * self.table.dimensions[0]
            angle = 0
        # on ramp
        elif place < 2 and place > 1:
            mag = (1 - place) * self.ramp.dimensions[0]
            angle = self.ramp_angle
        else:
            raise ValueError('Place not found')

        pos = pct_to_coord(mag, angle, z/2)
        obj.position = pos
        obj.orientation = (0, angle, 0)
        objects = self.objects
        objects[name] = obj
        self.objects = objects

    def serialize(self):
        d = {}
        d['ramp'] = self.ramp.serialize()
        d['table'] = self.table.serialize()
        d['objects'] = {k : o.serialize() for k,o in self.objects.items()}
        return d
