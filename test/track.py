#!/usr/bin/env python3
from rbw import simulation
from rbw.shapes import Ball
from rbw.utils.render import render
from galileo_ramp import TrackWorld, TrackSim

wrl = TrackWorld(10.0, 8.0, [0.1, 0.8, 0.146],
                 track_source = "/project/galileo_ramp/track/track.urdf")
# add objects here
ball_dims = [0.065, 0.065, 0.065]

def make_ball():
    return Ball('', ball_dims, {'density': 1.0,
                                'lateralFriction':1.0,
                                'rollingFriction':0.2,
                                'spinningFriction':0.2,
                                'resitution':1.0})
ball1 = make_ball()
ball2 = make_ball()
ball3 = make_ball()

wrl.add_object('1', ball1, 0.0, force = [0, -2.5, 0], vel = ([1, 0, 0], [0,0,0]))
# wrl.add_object('1', ball1, 0.0, )
wrl.add_object('2', ball2, 0.33)
wrl.add_object('3', ball3, 0.66)

data = wrl.serialize()
client = simulation.init_client(debug = False) # start a server
sim = simulation.init_sim(TrackSim, data, client) # load ramp into client
pla, rot, cols = simulation.run_full_trace(sim, debug=False,
                                           time_step = 1000, T = 10.0) # run simulation
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
blend_path = base_path + 'circle_track_small.blend'

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
