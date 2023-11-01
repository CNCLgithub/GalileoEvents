using Revise

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
    death_factor::Float64
end

function CPParams(client::Int64, objs::Vector{Int64},
    mprior::MaterialPrior, pprior::PhysPrior,
    event_concepts::Vector{Type{<:EventRelation}},
    obs_noise::Float64=0.,
    death_factor=10.)
    # configure simulator with the provided
    # client id
    sim = BulletSim(;client=client)
    # These are the objects of interest in the scene
    rigid_bodies = RigidBody.(objs)
    # Retrieve the default latents for the objects
    # as well as their initial positions
    # Note: alternative latents will be suggested by the `prior`
    template = BulletState(sim, rigid_bodies)

    CPParams(mprior, pprior, event_concepts, sim, template, length(objs), obs_noise, death_factor)
end

"""
Current state of the change point model, simulation state and event state
"""
struct CPState <: GMState
    bullet_state::BulletState
    active_events::Set{Int64}
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
@gen function cp_prior(params::CPParams)
    # initialize the kinematic state
    latents = params.template.latents
    params_filled = Fill(params, length(latents))
    new_latents = @trace(Gen.Map(cp_object_prior)(latents, params_filled), :objects)
    bullet_state = setproperties(params.template; latents = new_latents)

    # initialize the event state
    active_events = Set{Int64}()

    init_state = CPState(bullet_state, active_events)
    return init_state
end

"""
Bernoulli weight that event relation holds
"""
function predicate(t::Type{Collision}, a::RigidBodyState, b::RigidBodyState)
    if norm(Vector(a.linear_vel)-Vector(b.linear_vel)) < 0.01
        return 0
    end
    d = norm(Vector(a.position)-Vector(b.position))-0.175 # l2 distance
    clamp(exp(-15d), 0., 1.)
end

# TODO: surface distances, use pybullet (contact maybe not helpful)

# gen functional collections 

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
@gen function _collision_clause(pair_idx::Vector{Int64}, latents::Vector{BulletElemLatents})
    latents[pair_idx[1]] = @trace(update_latents(latents[pair_idx[1]]), :new_latents_a)
    latents[pair_idx[2]] = @trace(update_latents(latents[pair_idx[2]]), :new_latents_b)
    return latents
end

@gen function _no_event_clause(pair_idx, latents::Vector{BulletElemLatents})
    return latents
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

"""
function valid_relations(state::CPState, event_concepts::Vector{Type{EventRelation}})
    return event_concepts
    # TODO: replace by map in the end
    for EventRelation in event_concepts
        # TODO: decide if valid
    end
end

# map mcmc kernel (link: https://www.gen.dev/docs/stable/ref/mcmc/#Composing-Stationary-Kernels)
# set of proposal functions

# change randomly clause choices of mass 
# revise for which event, modify the categorical
# e.g. another event type for the same pair or another event for one event type


"""
iterate over event concepts and evaluate predicates to active/deactive
"""
@gen function event_kernel(state::CPState, params::CPParams)
    object_pairs = collect(combinations(state.bullet_state.kinematics, 2))
    pair_idx = repeat(collect(combinations(1:length(state.bullet_state.kinematics), 2)), length(params.event_concepts))
    pair_idx = [[0,0], pair_idx...] # for no event

    # map possible events to weight vector for birth decision using the predicates
    predicates = [predicate(event_type, a, b) for  event_type in params.event_concepts for (a, b) in object_pairs]

    # birth of one or no random event
    weights = copy(predicates)
    active_events = copy(state.active_events)
    for idx in active_events # active events should not be born again
        weights[idx-1] = 0
    end
    weights = [max(0, 1 - sum(weights)), weights...]
    # TODO: objects that are already involved in some events should not be involved in other event types as well
    weights_normalized = weights ./ sum(weights)
    
    # Draw born event
    events = vcat(NoEvent, [repeat([event_type], length(object_pairs)) for event_type in params.event_concepts]...)

    start_event_idx = @trace(categorical(weights_normalized), :start_event_idx)
    if start_event_idx > 1
        push!(active_events, start_event_idx)
    end

    switch = Gen.Switch(map(clause, events)...)

    updated_latents = @trace(switch(start_event_idx, pair_idx[start_event_idx], state.bullet_state.latents), :event)
    bullet_state = setproperties(state.bullet_state; latents = updated_latents)

    # death of one or no active event
    weights = [(idx+1 in active_events && idx+1 != start_event_idx) ? max(1. - predicates[idx] * params.death_factor, 0.) : 0.0 for idx in 1:length(predicates)] # dying has a much lower chance of being born

    weights = [max(0, 1-sum(weights)), weights...] # no event at index 1
    weights = weights ./ sum(weights) # normalize

    end_event_idx = @trace(categorical(weights), :end_event_idx)
    if end_event_idx > 1 # nothing happens when no event dies
        delete!(active_events, end_event_idx)
    end 

    return active_events, bullet_state
end

"""
for one object, observe the noisy position in every dimension
"""
@gen function observe_position(k::RigidBodyState, noise::Float64)
    pos = k.position
    # add noise to position
    obs = @trace(broadcasted_normal(pos, noise), :positions)
    return obs
end

"""
for one time step, run event and physics kernel
"""
@gen function kernel(t::Int, prev_state::CPState, params::CPParams)
    # event kernel
    active_events, bullet_state = @trace(event_kernel(prev_state, params), :events)

    # simulate physics for the next step based on the current state and observe positions
    bullet_state::BulletState = PhySMC.step(params.sim, bullet_state)
    obs = @trace(Gen.Map(observe_position)(bullet_state.kinematics, Fill(params.obs_noise, params.n_objects)), :observe)

    next_state = CPState(bullet_state, active_events)
    return next_state
end

"""
generate physical scene with changepoints in the belief state
"""
@gen function cp_model(t::Int, params::CPParams)
    # initalize the kinematic and event state
    init_state = @trace(cp_prior(params), :prior)
   
    # unfold the event and kinematic state over time
    states = @trace(Gen.Unfold(kernel)(t, init_state, params), :kernel)
    return states
end
