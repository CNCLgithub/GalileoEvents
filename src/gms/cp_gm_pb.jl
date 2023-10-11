
export CPParams,
    CPState,
    cp_model,
    EventRelation,
    Collision

using LinearAlgebra:norm
using Combinatorics

## Changepoint Model + components

"""
Event types
"""
abstract type EventRelation end

struct Collision <: EventRelation
    a::RigidBody
    b::RigidBody
end

struct NoEvent <: EventRelation end


"""
Parameters for change point model
"""
struct CPParams <: GMParams
    # prior
    material_prior::MaterialPrior
    physics_prior::PhysPrior
    # event relations
    event_concepts::Vector{Type{<:EventRelation}}
    # simulation
    sim::BulletSim
    template::BulletState
    n_objects::Int64
    obs_noise::Float64
end

function CPParams(client::Int64, objs::Vector{Int64},
    mprior::MaterialPrior, pprior::PhysPrior,
    event_concepts::Vector{Type{<:EventRelation}},
    obs_noise::Float64=0.)
    # configure simulator with the provided
    # client id
    sim = BulletSim(;client=client)
    # These are the objects of interest in the scene
    rigid_bodies = RigidBody.(objs)
    # Retrieve the default latents for the objects
    # as well as their initial positions
    # Note: alternative latents will be suggested by the `prior`
    template = BulletState(sim, rigid_bodies)

    CPParams(mprior, pprior, event_concepts, sim, template, length(objs), obs_noise)
end

"""
Current state of the change point model, simulation state and event state
"""
struct CPState <: GMState
    bullet_state::BulletState
    active_events::Vector{Int64}
end

## PRIOR

"""
initalizes prior beliefs about mass, friction and resitution of the given objects
"""
@gen function cp_object_prior(ls::RigidBodyLatents, gm::CPParams)
    # sample material
    mi = @trace(categorical(gm.material_prior.material_weights), :material)

    # sample physical properties
    phys_params = gm.physics_prior
    mass_mu, mass_sd = phys_params.mass
    mass = @trace(trunc_norm(mass_mu, mass_sd, 0., Inf), :mass)
    fric_mu, fric_sd = phys_params.friction
    friction = @trace(trunc_norm(fric_mu,fric_sd, 0., 1.), :friction)
    res_low, res_high = phys_params.restitution
    restitution = @trace(uniform(res_low, res_high), :restitution)

    # package
    new_ls = setproperties(ls.data;
                           mass = mass,
                           lateralFriction = friction,
                           restitution = restitution)
    new_latents::RigidBodyLatents = RigidBodyLatents(new_ls)
    return new_latents
end

"""
initializes belief about all objects and events
"""
@gen (static) function cp_prior(params::CPParams)
    # initialize the kinematic state
    latents = params.template.latents
    params_filled = Fill(params, length(latents))
    new_latents = @trace(Gen.Map(cp_object_prior)(latents, params_filled), :objects)
    bullet_state = setproperties(params.template; latents = new_latents)

    # initialize the event state
    active_events = Vector{EventRelation}()

    init_state = CPState(bullet_state, active_events)
    return init_state
end

"""
Bernoulli weight that event relation holds
"""
function predicate(t::Type{Collision}, a::RigidBodyState, b::RigidBodyState)
    d = norm(Vector(a.position)-Vector(b.position)) # l2 distance
    clamp(1.0 - d, 0., 1.)
end

"""
update latents of a single element
"""
@gen function update_latents(latents::BulletElemLatents)
    new_mass = @trace(trunc_norm(latents.data.mass, .1, 0., Inf), :mass)
    new_restitution = @trace(trunc_norm(latents.data.restitution, .1, 0., 1.), :restitution)

    new_latents = setproperties(latents.data;
                               mass = new_mass,
                               restitution = new_restitution)
    new_latents = RigidBodyLatents(new_latents)
    return new_latents
end



"""
in case of collision: Gaussian drift update of mass and restitution#
"""
@gen (static) function _collision_clause(pair_idx::Vector{Int64}, latents::Vector{<:BulletElemLatents})
    new_latents = @trace(update_latents(latents[pair_idx[1]]), :new_latents_a)
    new_latents = @trace(update_latents(latents[pair_idx[2]]), :new_latents_b)
    return Type{Collision}
end

@gen (static) function _no_event_clause(pair_idx, latents::Vector{BulletElemLatents})
    return Type{NoEvent}
end

"""
Returns the generative function over latents for the event relation
"""
function clause(::Type{Collision})
    _collision_clause
end

function clause(::Type{NoEvent})
    _no_event_clause
end

"""
objects that are already involved in some events should not be involved in new events
"""
function valid_relations(state::CPState, event_concepts::Vector{Type{EventRelation}})
    return event_concepts
    # TODO: replace by map in the end
    for EventRelation in event_concepts
        # TODO: decide if valid
    end
end

"""
iterate over event concepts and evaluate predicates to active/deactive
"""
@gen function event_kernel(state::CPState, event_concepts::Vector{Type{<:EventRelation}})
    # TODO: filter out invalid event relations (e.g., a collision is already active between a and b)
    #relations = valid_relations(state, event_concepts)

    # activate new events
    object_pairs = collect(combinations(state.bullet_state.kinematics, 2)) # get pairs of objects
    # map active events to weights 2D tensor for birth decision
    
    weights = [predicate(event_type, a, b) for  event_type in event_concepts for (a, b) in object_pairs]
    push!(weights, 1 - min(1, sum(weights))) # for no event
    weights = weights ./ sum(weights)
    #println(weights)

    pair_idx = repeat(collect(combinations(1:length(state.bullet_state.kinematics), 2)), length(event_concepts))
    push!(pair_idx, Vector([0,0])) # for no event
    #println(pair_idx)
    
    # draw one event overall
    event_idx = @trace(categorical(weights), :event_idx)
    active_events = [state.active_events..., event_idx]
    #push!(active_events, event_idx)
    #println(active_events)

    # Switch combinator to evaluate clauses for each event
    events = vcat([repeat([event_type], length(object_pairs)) for event_type in event_concepts]..., NoEvent)
    event = @trace(Gen.Switch(map(clause, events)...)(event_idx, pair_idx[event_idx], state.bullet_state.latents), :event)

    # TODO: some new events kill old events

    return active_events
end

"""
for one object, observe the noisy position in every dimension
"""
@gen (static) function observe_position(k::RigidBodyState, noise::Float64)
    pos = k.position
    # add noise to position
    obs = @trace(broadcasted_normal(pos, noise), :positions)
    return obs
end

"""
for one time step, run event and physics kernel
"""
@gen (static) function kernel(t::Int, prev_state::CPState, params::CPParams)
    # event kernel
    active_events = @trace(event_kernel(prev_state, params.event_concepts), :events)

    # simulate physics for the next step based on the current state and observe positions
    bullet_state::BulletState = PhySMC.step(params.sim, prev_state.bullet_state)
    obs = @trace(Gen.Map(observe_position)(bullet_state.kinematics, Fill(params.obs_noise, params.n_objects)), :observe)

    next_state = CPState(bullet_state, active_events)
    return next_state
end

"""
generate physical scene with changepoints in the belief state
"""
@gen (static) function cp_model(t::Int, params::CPParams)
    # initalize the kinematic and event state
    init_state = @trace(cp_prior(params), :prior)
   
    # unfold the event and kinematic state over time
    states = @trace(Gen.Unfold(kernel)(t, init_state, params), :kernel)
    return states
end
