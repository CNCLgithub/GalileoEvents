export Params,
    default_params,
    markov_generative_model

using PyCall

# world = physics.world
# physics = physics.world.simulation.physics
# object = physics.world.scene.shape
# ball = physics.world.scene.ball
# puck = physics.world.scene.puck
# block = physics.world.scene.block
# pyscene = physics.world.scene.ramp

## Structs and helpers
##
struct Params
    n_objects::Int
    object_prior::Vector{Dict}
end

const default_params = Params(2,
                              fill(Dict("shape" => "Block",
                                        "dims" => [0.3, 0.3, 0.15],
                                        "appearance" => nothing),
                                   2))
const density_map = Dict("Wood" => 1.0,
                         "Brick" => 2.0,
                         "Iron" => 8.0)
const friction_map = Dict("Wood" => 0.263,
                          "Brick" => 0.323,
                          "Iron" => 0.215)

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

function initialize_state(params::Params,
                          obj_phys,
                          init_pos)

    s = physics.scene.ramp.RampScene([3.5, 1.8], [3.5, 1.8],
                                     ramp_angle = 35. * pi / 180.)
    for i = 1:params.n_objects
        k = "$i"
        obj = create_object(params.object_prior[i], obj_phys[i])
        s.add_object(k, obj, init_pos[i])
    end
    scene = s.serialize()
    physics.physics.initialize_trace(scene)
end

function cleanup_state(client)
    physics.physics.clear_trace(client)
    return nothing
end

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


function forward_step(prev_state, client, obj_map)
    physics.physics.run_mc_trace(client, obj_map,
                         state = prev_state)
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

@gen (static) function kernel(t::Int, prev_state, client, obj_map)
    next_state = forward_step(prev_state, client, obj_map)
    pos = next_state[1]
    next_pos = @trace(Gen.broadcasted_normal(pos, 0.1),
                      :positions)
    return next_state
end

chain = Gen.Unfold(kernel)

@gen (static) function markov_generative_model(t::Int, params::Params)

    objects = @trace(map_object_prior(params.object_prior),
                     :object_physics)

    init_args = fill(tuple(), params.n_objects)
    initial_pos = @trace(map_init_state(init_args), :initial_state)
    phys_init = initialize_state(params, objects, initial_pos)
    states = @trace(chain(t, nothing, phys_init[1], phys_init[2]),
                    :chain)
    t = cleanup_state(phys_init[1])
    return states
end
