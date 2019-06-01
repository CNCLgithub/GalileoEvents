using Gen
using PyCall
using Rotations
using LinearAlgebra

include("dist.jl");
forward_model = pyimport("experiment.simulation.forward_model");

struct Scene
    blocks::Array
    unknown::Array{Int}
    mass_prior::Array{Float64}
end;


function quat_to_rotm(q::Matrix{Float64})
    h, w = size(q)
    mat = zeros((h, 9))
    for r = 1:h
        mat[r, :] = reshape(Quat(q[r, :]...), 1, 9)
    end
    return mat
end

@gen function simulate(tower::Dict, frames::Int)
    state = forward_model.simulate(tower, frames)
    pos = Matrix{Float64}(state["position"][end, :, :])
    rot = Matrix{Float64}(state["rotation"][end, :, :])
    rot = quat_to_rotm(rot)
    @trace(mat_noise(pos, 0.1), frames => :pos)
    @trace(mat_noise(rot, 0.1), frames => :rot)
    return (pos, rot, state)
end;

function make_tower(scene::Scene, unknown_mass::Float64, pos::Matrix{Float64})
    # don't count base block
    n_blocks = length(scene.blocks) - 1
    tower = deepcopy(scene.blocks)
    # assign positions
    for i = 1:n_blocks
        tower[i + 1]["data"]["pos"] = pos[i, :]
    end
    # assign masses
    for i in eachindex(scene.unknown)
        bid = scene.unknown[i]
        tower[bid]["data"]["substance"]["density"] = unknown_mass
    end
    return tower
end;

# @gen function mass_prior(scene::Scene)
#     n_blocks = length(scene.unknown)
#     masses = Vector{Float64}(undef, n_blocks)
#     for b_i in eachindex(scene.unknown)
#         masses[b_i] = @trace(normal(scene.base_mass, 3.5), scene.unknown[b_i])
#     end
#     return masses
# end;


@gen function init_positions(scene::Scene)
    n_blocks = length(scene.blocks) - 1
    positions = Matrix{Float64}(undef, n_blocks, 3)
    for i = 1:n_blocks
        # ignore base block
        block = scene.blocks[i + 1]
        pos = Vector{Float64}(undef, 3)
        pos[:] = block["data"]["pos"]
        new_x = @trace(normal(pos[1], 0.1), i => :x)
        new_y = @trace(normal(pos[2], 0.1), i => :y)
        positions[i, :] = [new_x, new_y, pos[3]]
    end
    return positions
end;

function debug_positions(scene::Scene)
    n_blocks = length(scene.blocks) - 1
    positions = Matrix{Float64}(undef, n_blocks, 3)
    for i = 1:n_blocks
        # ignore base block
        block = scene.blocks[i + 1]
        pos = Vector{Float64}(undef, 3)
        pos[:] = block["data"]["pos"]
        positions[i, :] = [pos[1], pos[2], pos[3]]
    end
    return positions
end;

function make_model(scene::Scene)::GenerativeFunction
    f = @gen function tower_model(nf::Int)
        # sample mass for blocks listed in scene.unknown
        unknown_mass = @trace(trunc_norm(scene.mass_prior...),
                              :unknown_mass)
        # add noise to initial positions for all blocks
        init_pos = debug_positions(scene)
        # init_pos = init_positions(scene)
        # init_pos = @trace(init_positions(scene), :init_pos)
        # setup tower using sampled values
        tower = make_tower(scene, unknown_mass, init_pos)
        # simulate, adding noise to resulting state-space
        state_space = @trace(simulate(tower, nf), :obs)
        # return noiseless state-space
        return state_space
    end;
    return f
end;


function make_obs(source; steps = 2)
    obs = Gen.choicemap()
    positions = source["position"]
    rotations = source["rotation"]
    n_frames = size(positions)[1]
    start = floor(n_frames / steps)
    frames = range(start, n_frames, length = steps)
    frames = Array{Int, 1}(collect(frames))
    for i in frames
        obs[:obs => i => :pos] = positions[i, :, :]
        obs[:obs => i => :rot] = quat_to_rotm(rotations[i, :, :])
    end
    return obs, frames
end;

function make_obs(model, gt; steps = 2)
    obs = Gen.choicemap()
    constraints = Gen.choicemap((:unknown_mass, gt))
    trace, _ = Gen.generate(model, (240,), constraints)
    _, _, source = get_retval(trace)
    positions = source["position"]
    rotations = source["rotation"]
    n_frames = size(positions)[1]
    start = floor(n_frames / steps)
    frames = range(start, n_frames, length = steps)
    frames = Array{Int, 1}(collect(frames))
    for i in frames
        obs[:obs => i => :pos] = positions[i, :, :]
        obs[:obs => i => :rot] = quat_to_rotm(rotations[i, :, :])
    end
    return obs, frames
end;
