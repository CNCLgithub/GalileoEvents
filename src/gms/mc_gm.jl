export Params,
    markov_generative_model

using PyCall


## Structs and helpers


const default_physics = Dict("density" => 2.0,
                             "lateralFriction" => 0.3)
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

# # for `test/markov_gm.jl`
# function Params(client::Int, object_map::Dict{String, Int})
#     n = length(object_map)
#     prior = fill(default_object, n)
#     Params(n, prior, 0.1, client, object_map)
# end
# during init
function Params(object_prior::Vector, init_pos::Vector{Float64},
                obs_noise::Float64, cid::Int)
    n = length(object_prior)
    scene = initialize_state(object_prior, init_pos)
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
    obj = shape("", params["dims"], physical_props)
end

function _init_state(object_prior::Vector,
                     object_phys,
                     init_pos)
    s = physics.scene.ramp.RampScene([3.5, 1.8], [3.5, 1.8],
                                     ramp_angle = 35. * pi / 180.)
    for i = 1:length(init_pos)
        k = "$i"
        obj = create_object(object_prior[i], object_phys[i])
        s.add_object(k, obj, init_pos[i])
    end
    scene::PyDict = s.serialize()
    # @pycall s.serialize()::PyDict
end

# for client init
function initialize_state(object_prior::Vector,
                          init_pos::Vector{Float64})
    phys = fill(default_physics, 2)
    _init_state(object_prior, phys, init_pos)
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
    for k in keys(objs)
        obj_d[k] = objs[k]["physics"]
    end
    return obj_d
end

# function cleanup_state(client)
#     physics.physics.clear_trace(client)
#     return nothing
# end

function from_material_params(params)
    mat = params["appearance"]
    if isnothing(mat)
        density_prior = (4., 3.)
        friction_prior = (0.3, 0.3)
    else
        density_prior = (density_map[mat]..., 2.0)
        friction_prior = (friction_map[mat]..., 0.3)
    end

    return Dict("density" => density_prior,
                "lateralFriction" => friction_prior)
end

function update_world(cid, obj_ids, scene)
    # pyscene = convert(PyDict{String, PyDict{String, Float64}}, scene)
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
    step(params.client, params.object_map, prev_state)
end

## Generative Model + components
##

@gen (static) function object_prior(material_params)
    mat_prop = from_material_params(material_params)
    dens_prior = mat_prop["density"]
    fric_prior = mat_prop["lateralFriction"]
    density = @trace(trunc_norm(dens_prior[1],
                                dens_prior[2],
                                0., 10.),
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
    physics_belief = initialize_state(params, objects, initial_pos)
    states = @trace(chain(t, nothing, params, physics_belief), :chain)
    # t = cleanup_state(phys_init[1])
    return states
end
