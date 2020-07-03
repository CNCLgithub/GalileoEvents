#!/usr/bin/env python3
from rbw import simulation
from galileo_ramp import TrackWorld, TrackSim

wrl = TrackWorld(10.0, 8.0, [0,0,0],
                 track_source = "/project/galileo_ramp/track/track.urdf")
data = wrl.serialize()
client = simulation.init_client(debug = True) # start a server
sim = simulation.init_sim(TrackSim, data, client) # load ramp into client
trace = simulation.run_full_trace(sim, T = 3.0, debug = True) # run simulation
simulation.clear_sim(sim)
