using Gen
using PyCall
using Rotations
using LinearAlgebra

include("dist.jl");
forward_model = pyimport("galileo_ramp.world.simulation.forward_model");

struct Scene
    data::Dict
    balls::Dict
    latents::Array{String, 1}
    nf::Int
    prior::Matrix{Float64}
end;

@gen function simulate(scene::Scene, latents::Dict, frames::Int)::Array{Float64,3}
    # Add the features from the latents to the scene descriptions
    data = copy(scene.data)
    for obj in keys(latents)
        merge!(data["objects"][obj], latents[obj])
    end
    state = forward_model.simulate(data, frames)
    pos = state[1]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, 0.1), frames => :pos)
    println(obs)
    return pos
end;


"""
Returns the GT positions for each ball.
"""
function debug_positions(scene::Scene)::Dict{String, Array{Float64}}
    n_balls = length(scene.balls)
    pos = Dict()
    for i in keys(scene.balls)
        pos[i] = scene.balls[i]["position"]
    end
    return pos
end;

"""
Returns a Gen function that serves as the generative model
that is parameterized by a given scene.

During inference, the random processes with the provided
addresses are update (resimulation in the case of MH).
"""
function make_model(scene::Scene)
    prior = scene.prior
    f = @gen function gm_model(t::Int)

        # add noise to initial positions for all blocks
        # init_pos = @trace(init_positions(scene), :init_pos)
        # init_pos = debug_positions(scene)

        # Sample physical latents and store then into a dict
        latents = Dict()
        for ball in keys(scene.balls)
            latents[ball] = Dict()
            for i = 1:length(scene.latents)
                l = scene.latents[i]
                v = @trace(uniform(prior[i, :]...), ball => l)
                latents[ball][l] = exp(v)
                # latents[ball]["position"] = init_pos[ball]
            end
        end

        # simulate, adding noise to resulting state-space
        st = @trace(simulate(scene, latents, t), :obs)

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
    positions = simulate(scene, Dict(), scene.nf)
    start = floor(scene.nf / steps)
    frames = range(start, scene.nf, length = steps)
    frames = Array{Int, 1}(collect(frames))
    obs = Gen.choicemap()
    for i in frames
        obs[:obs => i => :pos] = positions[i, :, :]
    end
    return obs, frames
end;
