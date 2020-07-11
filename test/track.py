#!/usr/bin/env python3
from rbw import simulation
from rbw.shapes import Ball
from rbw.utils.render import render
from galileo_ramp import TrackWorld, TrackSim

wrl = TrackWorld(10.0, 8.0, [1.0, 4.702, 4.0],
                 track_source = "/project/galileo_ramp/track/track.urdf")
# add objects here
ball1 = Ball('', [1., 1., 1.], {'density': 1.0, 'lateralFriction':0.5,
                                'resitution':0.9})
ball2 = Ball('', [1., 1., 1.], {'density': 1.0, 'lat4eralFriction':0.5,
                                'restitution':0.9})
ball3 = Ball('', [1., 1., 1.], {'density': 1.0, 'lat4eralFriction':0.5,
                                'restitution':0.9})

wrl.add_object('1', ball1, 1.5)
wrl.add_object('2', ball2, 0.5)
wrl.add_object('3', ball3, 0.76)

data = wrl.serialize()
client = simulation.init_client(debug = False) # start a server
sim = simulation.init_sim(TrackSim, data, client) # load ramp into client
pla, rot, cols = simulation.run_full_trace(sim, debug=False,  T = 3.0) # run simulation
simulation.clear_sim(sim)


trace = dict(
    pos = pla[:, 0],
    orn = rot,
    avl = pla[:, 1],
    lvl = pla[:, 2],
    col = cols
)

blender_exec = '/blender/blender'
base_path = '/project/galileo_ramp/track/'
render_path = base_path + 'render.py'
blend_path = base_path + 'track.blend'

kwargs = dict(
    scene = {'scene': data},
    trace = trace,
    out = 'test/track',
    render_mode = 'none',
    render = render_path,
    blend = blend_path,
    exec = blender_exec
)
render(**kwargs)
