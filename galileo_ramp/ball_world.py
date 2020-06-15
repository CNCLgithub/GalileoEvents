from rbw.worlds import RampWorld
from rbw.worlds.ramp import pct_to_coord

default_phys = {'lateralFriction': 0.2,
                'density': 0.0}

class Ball3World(RampWorld):
    """ 3Ball variant of ramp world.

    describes ramp marbles with possible inital velocities
    """
    def __init__(self, table_dims, ramp_dims, objects = None,
                    ramp_angle = 0.0, table_phys = default_phys,
                    ramp_phys = default_phys):

        super().__init__(table_dims, ramp_dims,
                         objects = objects, ramp_angle = ramp_angle,
                         table_phys = table_phys, ramp_phys = ramp_phys)
        self.init_vel = {}

    def add_object(self, name, obj, place,
                   init_vel = None):
        """
        Initial velocites should be a tuple (linear, angular)
        """
        z = obj.dimensions[-1]
        # on table
        if place < 1:
            mag = (1 - place) * self.table.dimensions[0]
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
        initial_pos = self.initial_pos
        initial_pos[name] = place
        self.initial_pos = initial_pos
        if not init_vel is None:
            self.init_vel[name] = init_vel

    def serialize(self):
        d = super().serialize()
        d['init_vel'] = self.init_vel
        return d
