from rbw.worlds import RampWorld

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
        super().add_object(name, obj, place)
        if not init_vel is None:
            self.init_vel[name] = init_vel


    def serialize(self):
        d = super().serialize()
        d['initial_vel'] = self.initial_vel
        return d
