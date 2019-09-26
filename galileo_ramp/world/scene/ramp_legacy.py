import numpy as np

from . import ramp, block


ramp_bounds = np.array([[  0.41577843,  31.58422089],
                        [-13.        ,   5.        ],
                        [  5.37593269,  14.62406731]])
table_bounds = np.array([[-34.5       ,   0.5       ],
                         [-12.90679741,   5.09320259],
                         [  4.69044113,   5.69044113]])

def pct_to_coord(obj, target, bounds, pct):
    '''
        Moves a given `bpy.object` along the ramp.
        Args:
            obj (bpy.Object): The object to move to the ramp.
            pct (float): The percentage of the incline to place. 0.0 is the
                bottom.
    '''

    dx = bounds[0][1] - bounds[0][0]
    dz = bounds[2][1] - bounds[2][0]

    m = dz / dx

    b = bounds[2][0] - (m * bounds[0][0])

    x = pct * target.dimensions[0] + bounds[0][0]

    z = m * x + b + (target.dimensions[2] + obj.dimensions[2]) / 2.
    return (x, target.position[1], z)

class LegacyRamp(ramp.RampScene):

    def __init__(self, table_dims, ramp_dims, objects = None,
                 ramp_angle = 0.0, table_friction = 0.8,
                 ramp_friction = 0.8):
        table = block.Block('table', (*table_dims, 1), 0, table_friction)
        # We want the table surface to be the xy plane @ z = 0
        table.position = (-17.0, -3.906797409057617, 5.190441131591797)
        # table.position = (0, 0, 0)
        self.table = table
        ramp = block.Block('ramp', (*ramp_dims, 1), 0, ramp_friction,
                                angle = (0, ramp_angle, 0))
        ramp.position = np.array((16, -4, 10))  # - np.array((-17.0, -3.91, 5.19))
        self.ramp = ramp
        self.ramp_angle = ramp_angle
        self.objects = objects

    def add_object(self, name, obj, place):
        z = obj.dimensions[-1]
        # on table
        if name == 'B':
            target = self.table
            bounds = table_bounds
            angle = 0
            # pos = pct_to_coord(obj, target, bounds, place)
            pos = [-4.75, -3.907, 6.44 ]
        # on ramp
        elif name == 'A':
            target = self.ramp
            bounds = ramp_bounds
            angle = self.ramp_angle
            # pos = pct_to_coord(obj, target, bounds, place)
            pos = [14.37,  -4., 11.249]
        else:
            raise ValueError('Place not found')

        obj.position = pos
        obj.orientation = (0, angle, 0)
        objects = self.objects
        objects[name] = obj
        self.objects = objects
