
export GMParams,
    Params,
    create_object,
    initialize_state,
    forward_step


using PyCall


"""Parameters for using the Bullet physics engine"""
@with_kw struct BulletGM <: PhysicsGM
    n_objects::Int64 = 2
    object_prior::Vector{BulletPrior}
    obs_noise::Float64
    client::Int64
    object_map::Dict{String, Int64}
end

"""

State for `BulletGM`.

# Properties
- client: The bullet client id
- latents: The physical properties for each object

"""
struct BulletState <: SimState
    # simulation context
    client::Int64
    latents::Dict{String, BulletLatents}
    kinematics::BulletKinematics
end


@with_kw struct BulletLatents
    density::Float64 = 1.0
    lateral_friction::Float64 = 1.0
    volume::Float64 = 1.0
    mass::Float64 = density * volume
end

struct BulletKinematics
    positions::PyArray
    orientations::PyArray
    linear_velicities::PyArray
    angular_velocities::PyArray
end

const surface_latents = BulletLatents(
    density = 0.0, # static object
    lateral_friction = 0.5,
)

abstract type Material end

abstract type BulletMaterial <: Material end

struct Wood <: BulletMaterial end
const wood = Wood()
prior(wood) = (density = 1.0,
               lateral_friction = 0.263)

struct Brick <: BulletMaterial end
const brick = Brick()
prior(Brick) = (density = 2.0,
                lateral_friction = 0.323)


struct Iron <: BulletMaterial end
const iron = Iron()
prior(Iron) = (density = 8.0,
               lateral_friction = 0.215)

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
const mat_keys = collect(keys(density_map))
const default_prior_width  = 0.1

abstract type GMParams end

struct Params <: GMParams
    # during init
    function Params(object_prior::Vector, init_pos::Vector{Float64},
                    scene::Dict, obs_noise::Float64, prior_width::Float64,
                    cid::Int)
        n = length(object_prior)
        object_map = @pycall physics.physics.init_world(scene,
                                                        cid)::Dict{String,Int}
        new(n, object_prior, obs_noise, prior_width, cid, object_map)
    end
end


function create_object(params, physical_props)
    cat = params["shape"]
    if cat == "Block"
        shape = physics.world.Block
    elseif cat == "Puck"
        shape = physics.world.Puck
    else
        shape = physics.world.Ball
    end
    obj::PyObject = shape("", params["dims"], physical_props)
end

function update_object!(physical_props, params)
    cat = params["shape"]
    density = physical_props["density"]
    volume = params["volume"]
    physical_props["mass"] = density * volume
    return nothing
end

function _init_state(object_prior::Vector,
                     object_phys,
                     init_pos)
    s::PyObject = physics.world.RampScene([3.5, 1.8], [3.5, 1.8],
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

# for inference
obj_phys_type = Dict{String, Float64}
function initialize_state(params::Params,
                          object_phys,
                          init_pos)
    scene = _init_state(params.object_prior, object_phys,
                        init_pos)
    objs = scene["objects"]
    objs["B"]["physics"]["persistent"] = true
    obj_v = Vector{obj_phys_type}(undef, params.n_objects)
    init_mat = zeros(4, 2, 3)
    for (o,k) in enumerate(["A", "B"])
        params.object_prior[o] = objs[k]
        obj_v[o] = objs[k]["physics"]
        init_mat[1,o,:] = objs[k]["position"]
        init_mat[2,o,:] = objs[k]["orientation"]
    end
    graph = (false, (false, false))
    return (init_mat, graph, obj_v)
end



function update_world!(belief,
                       params::Params)
    scene = Dict{String, Any}()
    for (i,k) = enumerate(["A", "B"])
        update_object!(belief[i], params.object_prior[i])
        scene[k] = belief[i]
    end
    @pycall physics.physics.update_world(params.client,
                                         params.object_map,
                                         scene)::PyObject
    return nothing
end

function step(params::Params, prev_state)
    @pycall physics.physics.run_mc_trace(params.client,
                                         params.object_map,
                                         state = prev_state)::Array{Float64,3}
end

function forward_step(prev_state, params::Params,
                      belief)
    update_world!(belief, params)
    state = step(params, prev_state)
    return state
end
