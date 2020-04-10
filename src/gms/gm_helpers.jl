
export GMParams,
    Params,
    create_object,
    initialize_state,
    from_material_params,
    forward_step


using PyCall


const surface_phys = Dict("density" => 0,
                          "lateralFriction" => 0.5)
const default_physics = Dict("density" => 2.0,
                             "lateralFriction" => 0.3,
                             "restitution" => 0.9)
const default_object = Dict("shape" => "Block",
                            "dims" => [0.3, 0.3, 0.15],
                            "appearance" => nothing,
                            "physics" => default_physics)
const density_map = Dict("Wood" => 1.0,
                         "Brick" => 2.0,
                         "Iron" => 8.0)
const friction_map = Dict("Wood" => 0.263,
                          "Brick" => 0.323,
                          "Iron" => 0.215)

abstract type GMParams end

struct Params <: GMParams
    n_objects::Int
    object_prior::Vector{Dict}
    obs_noise::Float64
    client::Int
    object_map::Dict{String, Int}
end

# during init
function Params(object_prior::Vector, init_pos::Vector{Float64},
                scene::Dict,
                obs_noise::Float64, cid::Int)
    n = length(object_prior)
    object_map = @pycall physics.physics.init_world(scene, cid)::Dict{String,Int}
    Params(n, object_prior, obs_noise, cid, object_map)
end

function create_object(params, physical_props)
    cat = params["shape"]
    if cat == "Block"
        shape = physics.scene.block.Block
    elseif cat == "Puck"
        shape = physics.scene.puck.Puck
    else
        shape = physics.scene.ball.Ball
    end
    obj::PyObject = shape("", params["dims"], physical_props)
end

function _init_state(object_prior::Vector,
                     object_phys,
                     init_pos)
    s::PyObject = physics.scene.ramp.RampScene([3.5, 1.8], [3.5, 1.8],
                                               ramp_angle = 35. * pi / 180.,
                                               ramp_phys = surface_phys,
                                               table_phys = surface_phys)

    for (i,k) = enumerate(["A", "B"])
        obj::PyObject = create_object(object_prior[i], object_phys[i])
        s.add_object(k, obj, init_pos[i])
    end
    scene::PyDict = s.serialize()
end

# for Params init
function initialize_state(object_prior::Vector,
                          init_pos::Vector{Float64})
    phys = [d["physics"] for d in object_prior]
    _init_state(object_prior, phys, init_pos)::Dict
end
function initialize_state(scene::Dict)
    new_d = Dict(
        "ramp" => scene["ramp"],
        "table" => scene["table"],
        "initial_pos" => scene["initial_pos"],
        "objects" => Dict(
            "1" => scene["objects"]["A"],
            "2" => scene["objects"]["B"],
        )
    )
end

# for inference
obj_phys_type = Dict{String, Float64}
function initialize_state(params::Params,
                          object_phys,
                          init_pos)
    scene = _init_state(params.object_prior, object_phys,
                        init_pos)
    objs = scene["objects"]
    obj_v = Vector{obj_phys_type}(undef, params.n_objects)
    init_mat = zeros(4, 2, 3)
    for (o,k) in enumerate(["A", "B"])
        obj_v[o] = objs[k]["physics"]
        init_mat[1,o,:] = objs[k]["position"]
        init_mat[2,o,:] = objs[k]["orientation"]
    end
    return (init_mat, obj_v)
end

function from_material_params(params)
    mat = params["appearance"]
    if isnothing(mat)
        density_prior = (4., 4.)
        friction_prior = (0.3, 0.4)
    else
        density_prior = (density_map[mat]..., 4.0)
        friction_prior = (friction_map[mat]..., 0.4)
    end

    return Dict("density" => density_prior,
                "lateralFriction" => friction_prior)
end

function update_world(cid, obj_ids, belief)
    scene = Dict{String, obj_phys_type}()
    for (o,k) in enumerate(["A", "B"])
        scene[k] = belief[o]
    end
    @pycall physics.physics.update_world(cid,
                                         obj_ids,
                                         scene)::PyObject
    return nothing
end

function step(cid, obj_ids, prev_state)
    @pycall physics.physics.run_mc_trace(cid,
                                         obj_ids,
                                         state = prev_state)::Array{Float64,3}
end

function forward_step(prev_state, params::Params,
                      belief)
    update_world(params.client, params.object_map, belief)
    state = step(params.client, params.object_map, prev_state)
    return state
end
