import .


def pct_to_coord(mag, angle, z):
    z_offset = (np.sin(angle) * abs(mag)) + (np.cos(angle) * z)
    x_offset = (np.cos(angle) * mag) - (np.sin(angle) * z)
    return np.array((x_offset, 0, z_offset))

class RampScene:

    """
    Describes a ramp scene
    """

    def __init__(self, table_dims, ramp_dims, objects = None,
                 ramp_angle = 15.0):
        self.table = block.Block('table', table_dims, 0, table_friction)
        self.ramp = block.Block('ramp', ramp_dims, 0, ramp_friction,
                                rot = ramp_angle)
        self.objects = objects

    @property
    def objects(self):
        if self._objects is None:
            return {}
        else:
            return self._objects

    @objects.setter
    def objects(self, v):
        self._objects = v

    def add_object(name, obj, place):
        z = obj.dimensions[-1]
        if place < 1:
            mag = place * self.table.dimensions[0]
            coords = pct_to_coord(mag, 0, z)
        elif place < 2:
            mag = (1 - place) * self.ramp.dimensions[0]
            coords = pct_to_coord(mag, self.ramp_angle, z)
        else:
            raise ValueError('Place not found')

        objects = self.objects
        objects[name] = (coords, obj)
        self.objects = objects

    def serialize(self):
        d = {}
        d['ramp'] = self.ramp.serialize()
        d['table'] = self.table.serialize()
        d['objects'] = {k : {'pct':p, **o.serialize()}
                        for k,(p,o) in self.objects}
        return d
