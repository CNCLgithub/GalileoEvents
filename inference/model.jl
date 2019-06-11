using Gen
using PyCall
using Rotations
using LinearAlgebra

include("dist.jl");
forward_model = pyimport("experiment.simulation.forward_model");

struct Scene
    balls::Array{Symbol}
    data::Dict{Symbol}
end;

@gen function simulate(scene::Scene, frames::Int, latents::Dict)::Matrix{Float64}
    # Add the features from the latents to the scene descriptions
    data = merge(scene.data, latents)
    state = forward_model.simulate(data, frames)
    pos = state[0]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, scene.obs_noise), frames => :pos)
    return obs
end;


"""
Returns the GT positions for each ball.
"""
function debug_positions(scene::Scene)
    n_balls = length(scene.balls)
    pos = Dict()
    for i in scene.balls
        pos[i] = scene.data[i]['pos']
    end
    return pos
end;

"""
Returns a Gen function that serves as the generative model
that is parameterized by a given scene.

During inference, the random processes with the provided
addresses are update (resimulation in the case of MH).
"""
function make_model(scene::Scene, prior::Array{Float64})
    f = @gen function gm_model()

        # add noise to initial positions for all blocks
        # init_pos = @trace(init_positions(scene), :init_pos)
        init_pos = debug_positions(scene)

        # Sample physical latents and store then into a dict
        latents = Dict()
        for ball in scene.balls:
            for i = 1:length(scene.latents)
                l = scene.latents[i]
                latents[ball => l] = \
                    @trace(trunc_norm(prior[i]...), ball => l)
                latents[ball => :pos] = init_pos[ball]
            end
        end

        # simulate, adding noise to resulting state-space
        st = @trace(simulate(scene, latents), :obs)

        # The GM does not need to return anything but
        # feel free to edit this for QOL during updating.
        return st
    end;
    return f
end;

"""
Returns a choicemap for the GT of a given scene
"""
function make_obs(scene::Scene; steps = 2)
    positions = forward_model.simulate(scene.data, scene.nf)
    start = floor(nf / steps)
    frames = range(start, nf, length = steps)
    frames = Array{Int, 1}(collect(frames))
    for i in frames
        obs[:obs => i => :pos] = positions[i, :, :]
    end
    return obs, frames
end;
