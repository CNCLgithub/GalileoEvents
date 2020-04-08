export Params,
    markov_generative_model

using PyCall


## Structs and helpers

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

struct Params
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
    # scene = initialize_state(scene)
    # scene = initialize_state(object_prior, init_pos)
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
    obj_d = Dict{String, obj_phys_type}()
    init_mat = zeros(4, 2, 3)
    for (o,k) in enumerate(["A", "B"])
        obj_d[k] = objs[k]["physics"]
        init_mat[1,o,:] = objs[k]["position"]
        init_mat[2,o,:] = objs[k]["orientation"]
    end
    return (init_mat, obj_d)
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

function update_world(cid, obj_ids, scene)
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

function forward_step(prev_state, params::Params, belief::Dict)
    update_world(params.client, params.object_map, belief)
    state = step(params.client, params.object_map, prev_state)
    return state
end

## Generative Model + components
##

@gen (static) function object_prior(material_params)
    mat_prop = from_material_params(material_params)
    dens_prior = mat_prop["density"]
    fric_prior = mat_prop["lateralFriction"]
    density = @trace(trunc_norm(dens_prior[1],
                                dens_prior[2],
                                0., 150.),
                     :density)
    friction = @trace(trunc_norm(fric_prior[1],
                                 fric_prior[2],
                                 0., 1.),
                      :friction)
    restitution = @trace(uniform(0.8, 1.0), :restitution)
    physical_props = Dict("density" => density,
                          "lateralFriction" => friction,
                          "restitution" => restitution)
    return physical_props
end

map_object_prior = Gen.Map(object_prior)

@gen (static) function state_prior()
    init_pos = @trace(uniform(0, 2), :init_pos)
    return init_pos
end

map_init_state = Gen.Map(state_prior)

@gen (static) function kernel(t::Int, prev_state, params, belief)
    next_state = forward_step(prev_state, params, belief)
    pos = next_state[1, :, :]
    next_pos = @trace(Gen.broadcasted_normal(pos, params.obs_noise),
                      :positions)
    return next_state
end

chain = Gen.Unfold(kernel)

@gen (static) function markov_generative_model(t::Int, params::Params)

    objects = @trace(map_object_prior(params.object_prior),
                     :object_physics)
    init_args = fill(tuple(), params.n_objects)
    initial_pos = @trace(map_init_state(init_args), :initial_state)
    i_state = initialize_state(params, objects, initial_pos)

    states = @trace(chain(t, i_state[1], params, i_state[2]), :chain)
    return states
end
