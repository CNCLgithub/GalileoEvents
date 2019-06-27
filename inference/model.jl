using Gen
using PyCall
using Rotations
using LinearAlgebra
using DataStructures

include("dist.jl");
forward_model = pyimport("galileo_ramp.world.simulation.forward_model");

struct Scene
    data::Dict
    balls::Dict
    latents::Array{String, 1}
    nf::Int
    prior::Matrix{Float64}
end;

"""
Defines at which frames objects become active.
"""
function get_active(linear_vels::Array{Float64, 3})::Array{Int,1}
    n_frames, n_objs = size(linear_vels)[1:end-1]
    active_map = Array{Int, 1}(undef, n_objs)
    vels = sum(abs(linear_vels), axis = 2)
    for c = 1:n_objs
        active_map[c] = findfirst(vels[:, c] .> 0)
    end
    return active_map
end

@gen function simulate(scene::Scene, latents::Dict, frames::Int)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    for obj in keys(latents)
        obj_data = data["objects"][obj]
        merge!(obj_data, latents[obj])
        # TODO: automate this, replace with "volume"
        obj_data["mass"] = obj_data["density"] * obj_data["volume"]
        merge!(data["objects"][obj], obj_data)
    end
    state = forward_model.simulate(data, frames)
    pos = state[1]
    lin_vels = state[2]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, 0.1), frames => :pos)
    # println(obs)
    return pos,lin_vels
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
    f = @gen function gm_model(t::Int, active::Array{Int})

        # add noise to initial positions for all blocks
        # init_pos = @trace(init_positions(scene), :init_pos)
        # init_pos = debug_positions(scene)

        # Sample physical latents and store then into a dict
        latents = OrderedDict({Pair{String, String}, Float64})
        for i = 1:length(scene.latents)
            l = scene.latents[i]
            for j,obj in enumerate(scene.balls)
                if t >= active[j]
                    v = @trace(uniform(prior[i, :]...), obj => l)
                else
                    v = uniform(prior[i, :]...)
                end
                latents[obj => l] = exp(v)
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
    positions, active_map = simulate(scene, Dict(), scene.nf)
    start = floor(scene.nf / steps)
    frames = range(start, scene.nf, length = steps)
    frames = Array{Int, 1}(collect(frames))
    obs = Gen.choicemap()
    epoch = Array{Bool, 2}(false, steps, positions.shape[2])
    for i in frames
        obs[:obs => i => :pos] = positions[i, :, :]
    end
    return obs, active_map
end;
