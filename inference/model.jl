using Gen
using PyCall
using Rotations
using LinearAlgebra

include("dist.jl");
forward_model = pyimport("galileo_ramp.world.simulation.forward_model");

struct Scene
    data::Dict
    balls::Array{String, 1}
    latents::Array{String, 1}
    nf::Int
    prior::Matrix{Float64}
end;

"""
Defines at which frames objects become active.
"""
function get_active(objs::Array{String},
                    linear_vels::Array{Float64, 3})::Array{String,2}
    n_frames, n_objs = size(linear_vels)[1:end-1]
    vels = sum(abs.(linear_vels), dims = 3)
    active_map = fill("", (n_frames, n_objs))
    for c = 1:n_objs
        t = findfirst(vels[:, c] .> 1e-2)
        active_map[t:end, c] = fill(objs[c], n_frames-t+1)
    end
    return active_map
end

"""
Runs the forward model on the given scene,
replacing any features found in `latents`
"""
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
    state = forward_model.simulate(data, frames, objs = scene.balls)
    pos = state[1]
    lin_vels = state[4]
    obs = Matrix{Float64}(pos[end, :, :])
    @trace(mat_noise(obs, 0.1), frames => :pos)
    return pos,lin_vels
end;


"""
Returns the GT positions for each ball.
"""
function debug_positions(scene::Scene)::Dict{String, Array{Float64}}
    n_balls = length(scene.balls)
    pos = Dict()
    for i in keys(scene.balls)
        pos[i] = scene.data["objects"][i]["position"]
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
    f = @gen function gm_model(t::Int, active::Array{String})

        # add noise to initial positions for all blocks
        # init_pos = @trace(init_positions(scene), :init_pos)
        # init_pos = debug_positions(scene)

        # Sample physical latents and store then into a dict
        latents = Dict{String, Dict{String, Float64}}()
        for (j,obj) in enumerate(scene.balls)
            obd = Dict{String, Float64}()
            for i = 1:length(scene.latents)
                l = scene.latents[i]
                if obj == active[j]
                    v = @trace(uniform(prior[i, :]...), obj => l)
                else
                    v = uniform(prior[i, :]...)
                end
                obd[l] = exp(v)
            end
            latents[obj] = obd
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
and a list of objects to track for each observation.
"""
function make_obs(scene::Scene; ts::Array{Int, 1},
                  factorize::Bool = false)
    positions, lin_vel = simulate(scene, Dict(), ts[end])
    if factor
        active_map = get_active(scene.balls, lin_vel)[ts, :]
        args = collect(zip(ts, eachrow(active_map)))
    else
        active_map = repeat(scene.balls, 1, steps)
        args = collect(zip(ts, eachcol(active_map)))
    end
    obs = Gen.choicemap()
    for t in ts
        obs[:obs => t => :pos] = positions[t, :, :]
    end
    return obs, args
end;
