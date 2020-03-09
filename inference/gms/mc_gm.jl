using Gen
using PyCall

include("../distributions.jl")
world = pyimport("physics.world");


physics = pyimport("physics.world.simulation.physics")
object = pyimport("physics.world.scene.shape")
ball = pyimport("physics.world.scene.ball")
puck = pyimport("physics.world.scene.puck")
block = pyimport("physics.world.scene.block")
pyscene = pyimport("physics.world.scene.ramp")

## Structs and helpers
##
struct Params
    n_objects::Int
    object_prior::Vector{Dict}
end

const default_params = Params(2,
                              fill(Dict("shape" => "Block",
                                        "dims" => [2,2,2]), 2))

obj_type = typeof(object)
function initialize_state(params::Params,
                          obj_phys,
                          init_pos)

    s = pyscene.default_scene()
    for i = 1:params.n_objects
        k = "$i"
        obj = create_object(params.object_prior[i], obj_phys[i])
        s.add_object(k, obj, init_pos[i])
    end
    initial_state = nothing
    return (initial_state, s.serialize())
    # scene = physics.RampPhysics(s.serialize())
    # return (initial_state, scene)
end

function from_material_params(params)
    # TODO: add material prior
    density_prior = (4., 3.)
    friction_prior = (0.3, 0.3)
    return Dict("density" => density_prior,
                "lateralFriction" => friction_prior)
end

function create_object(params, physical_props)
    cat = params["shape"]
    if cat == "Block"
        shape = block.Block
    elseif cat == "Puck"
        shape = puck.Puck
    else
        shape = ball.Ball
    end
    obj = shape("", params["dims"], physical_props)
    # obj_data = obj.serialize()
end
function forward_step(prev_state, s)

    objects = s["objects"]
    obj_names = ["$x" for x in 1:length(objects)]
    trace  = physics.run_mc_trace(s,
                                  obj_names,
                                  state = prev_state,
                                  fps = 30,
                                  time_scale = 10.0)
    return trace
end
# function forward_step(prev_state, scene)

#     fps = 30.0
#     objects = ["$x" for x in 1:length(scene.world)]
#     trace =  scene.get_trace(1. / fps, objects,
#                              state = prev_state,
#                              fps = 30,
#                              time_scale = 10.0)
#     return [t[1, :, :] for t in trace]
# end

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
    # obj = create_object(material_params, physical_props)
    return physical_props
end

map_object_prior = Gen.Map(object_prior)

@gen (static) function state_prior()
    init_pos = @trace(uniform(0, 2), :init_pos)
    return init_pos
end

map_init_state = Gen.Map(state_prior)

@gen (static) function kernel(t::Int, prev_state, scene)
    next_state = forward_step(prev_state, scene)
    pos = next_state[1]
    next_pos = @trace(Gen.broadcasted_normal(pos, 0.1),
                      :positions)
    return next_state
end

chain = Gen.Unfold(kernel)

@gen (static) function generative_model(t::Int, params::Params)

    objects = @trace(map_object_prior(params.object_prior),
                     :object_physics)

    init_args = fill(tuple(), params.n_objects)
    initial_pos = @trace(map_init_state(init_args), :initial_state)
    init_state = initialize_state(params, objects, initial_pos)
    states = @trace(chain(t, init_state[1], init_state[2]), :chain)
    return states
end

function test(n::Int)
    Gen.load_generated_functions()
    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = 1.5
    cm[:initial_state => 2 => :init_pos] = 0.5
    cm[:object_physics => 1 => :friction] = 0.2
    cm[:object_physics => 2 => :friction] = 0.2


    trace, w = Gen.generate(generative_model, (n, default_params), cm)
    println(get_choices(trace))
    return get_retval(trace)
end
