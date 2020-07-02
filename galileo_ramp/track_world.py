from rbw import np
from rbw.worlds import World
from rbw.shapes import Shape, Block

default_phys = {
        'lateralFriction': 0.2,
        'density': 0.0
        }

MeshShape = Shape()

class TrackWorld(World):
    """
    Describes a track scene
    """

    def __init__(self, outer_track_radius, inner_track_radius, ramp_dims, objects = None,
                ramp_angle = 43.0, ramp_phys = default_phys, track_dims = track_dims, track_phys = default_phys)
        ramp = Block('ramp', ramp_dims, ramp_phys, angle = (0, ramp_angle, 0))
        ramp.position = [0, -4.2582, 3.2272]
        track = MeshShape()
        # establish track positions
    
    def add_object(self, name, obj, place):
        z = obj.dimensions[-1]
        # on track
        # on ramp
