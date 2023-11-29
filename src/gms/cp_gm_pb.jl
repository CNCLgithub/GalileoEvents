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
holds parameters for change point model
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

"""
constructs parameter struct for change point model
"""
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

    a_dim = a.aabb[2] - a.aabb[1]
    b_dim = b.aabb[2] - b.aabb[1]
    d = norm(Vector(a.position)-Vector(b.position))-norm((a_dim+b_dim)/2) # l2 distance
    clamp(exp(-15d), 1e-3, 1 - 1e-3)
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
in case of collision: Gaussian drift update of mass and restitution
"""
@gen function _collision_clause(pair_idx::Vector{Int64}, latents::Vector{BulletElemLatents})
    latents[pair_idx[1]] = @trace(update_latents(latents[pair_idx[1]]), :new_latents_a)
    latents[pair_idx[2]] = @trace(update_latents(latents[pair_idx[2]]), :new_latents_b)
    return latents
end

function clause(::Type{Collision})
    _collision_clause
end

"""
in case of no event: no change
"""
@gen function _no_event_clause(pair_idx, latents::Vector{BulletElemLatents})
    return latents
end

function clause(::Type{NoEvent})
    _no_event_clause
end

event_concepts = Type{<:EventRelation}[NoEvent, Collision]
switch = Gen.Switch(map(clause, event_concepts)...)

"""
TODO: this function was intended to check if some event relations are impossible to be created a certain time step
"""
function valid_relations(state::CPState, event_concepts::Vector{Type{EventRelation}})
    return event_concepts
    # TODO: replace by map in the end
    for EventRelation in event_concepts
        # TODO: decide if valid
    end
end

@gen function event_switch(clause, events, start_event_idx, pair, latents)
    switch = Gen.Switch(map(clause, events)...)
    return switch(start_event_idx, pair, latents)
end

"""
map possible events to weight vector for birth decision using the predicates
"""
function calculate_predicates(obj_states)
    object_pairs = collect(combinations(obj_states, 2))
    pair_idx = repeat(collect(combinations(1:length(obj_states), 2)), length(event_concepts))
    pair_idx = [[0,0], pair_idx...] # [0,0] for no event

    # break up to two lines
    predicates = [predicate(event_type, a, b) for event_type in event_concepts for (a, b) in object_pairs if event_type != NoEvent] # NoEvent excluded and added in weights
    event_ids = vcat(1, repeat(2:length(event_concepts), inner=length(object_pairs))) # 1 for NoEvent
    return predicates, event_ids, pair_idx
end

"""
transform predicates for pairs of objects into a probability vector that adds to 1, including one weight for NoEvent at the first position
"""
function normalize_weights(weights, active_events)
    for idx in active_events # active events should not be born again
        weights[idx-1] = 0
    end
    weights = [max(0, 1 - sum(weights)), weights...] # first element for NoEvent
    # TODO: objects that are already involved in some events should not be involved in other event types as well
    return weights ./ sum(weights)
end

"""
similar to normalize_weights but for death of events
"""
function calculate_death_weights(predicates, active_events, start_event_idx, death_factor)
    can_die(idx) = idx+1 in active_events && idx+1 != start_event_idx
    # dying has a much lower chance of being born
    get_weight(idx) = can_die(idx) ? max(1. - predicates[idx] * death_factor, 0.) : 0.0  
    weights = [get_weight(idx) for idx in 1:length(predicates)]
    weights = [max(0, 1-sum(weights)), weights...] # no event at index 1
    return weights ./ sum(weights)
end

"""
updates active events in a functional form
add=True -> add event to set of active events
add=False -> remove event from set of active events
"""
function update_active_events(active_events::Set{Int64}, event_idx::Int64, add::Bool) 
    if event_idx == 1
        return active_events
    end
    if add
        return union(active_events, Set([event_idx]))
    else
        return setdiff(active_events, Set([event_idx])) 
    end
end


"""
iterate over event concepts and evaluate predicates for newly activated events
"""
@gen function event_kernel(active_events, bullet_state, death_factor)
    predicates, event_ids, pair_idx = calculate_predicates(bullet_state.kinematics)
    weights = normalize_weights(copy(predicates), active_events)
    start_event_idx = @trace(categorical(weights), :start_event_idx) # up to one event is born

    updated_latents = @trace(switch(event_ids[start_event_idx], pair_idx[start_event_idx], bullet_state.latents), :event)
    bullet_state = setproperties(bullet_state; latents = updated_latents)
    active_events = update_active_events(active_events, start_event_idx, true)
    
    weights = calculate_death_weights(predicates, active_events, start_event_idx, death_factor)
    end_event_idx = @trace(categorical(weights), :end_event_idx) # up to one active event dies
    active_events = update_active_events(active_events, end_event_idx, false)

    return active_events, bullet_state
end

"""
for one object, observe the noisy position in every dimension
"""
@gen function observe_position(k::RigidBodyState, noise::Float64)
    @trace(broadcasted_normal(k.position, noise), :positions)
end

"""
run event and physics kernel for one time step and observe noisy positions
"""
@gen function kernel(t::Int, prev_state::CPState, params::CPParams)
    active_events, bullet_state = @trace(event_kernel(prev_state.active_events,
                                                      prev_state.bullet_state,
                                                      params.death_factor), :events)

    bullet_state::BulletState = PhySMC.step(params.sim, bullet_state)
    @trace(Gen.Map(observe_position)(bullet_state.kinematics, Fill(params.obs_noise, params.n_objects)), :observe)

    return CPState(bullet_state, active_events)
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
