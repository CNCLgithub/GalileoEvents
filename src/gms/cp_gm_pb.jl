
export CPParams,
    CPState,
    cp_model

using LinearAlgebra:norm

## Changepoint Model + components

"""
Event types
"""
abstract type EventRelation end

struct Collision <: EventRelation
    a::RigidBody
    b::RigidBody
end


"""
Parameters for change point model
"""
struct CPParams <: GMParams
    # prior
    material_prior::MaterialPrior
    physics_prior::PhysPrior
    # event relations
    event_concepts::Vector{Type{EventRelation}}
    # simulation
    sim::BulletSim
    template::BulletState
    n_objects::Int64
    obs_noise::Float64
end

"""
Current state of the change point model, simulation state and event state
"""
struct CPState <: GMState
    bullet_state::BulletState
    active_events::Vector{EventRelation}
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
    active_events = Vector{EventRelation}()

    init_state = CPState(bullet_state, active_events)
    return init_state
end

"""
Bernoulli weight that event relation holds
"""
function predicate(Type{Collision}, x::RigidBodyState, y::RigidBodyState)
    d = norm(x.position, y.position) # l2 distance
    clamp(1.0 - d, 0., 1.)
end

"""
in case of collision: Gaussian drift update of mass and restitution#
"""
@gen static function _collision_clause(event_idx, latents::RigidBodyLatents)
    new_mass = @trace(trunc_norm(latents.mass, .1, 0., Inf), :mass)
    new_restitution = @trace(trunc_norm(latents.restitution, .1, res_low, res_high), :restitution)
    return set_properties(latents; mass = new_mass, restitution = new_restitution)
end

"""
Returns the generative function over latents for the event relation
"""
function clause(Type{Collision})
    _collision_clause
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


@gen function update_latents(state, latents)
    new_latents = setproperties(latents.data;
                               mass = mass,
                               lateralFriction = friction,
                               restitution = restitution)
    new_latents::RigidBodyLatents = RigidBodyLatents(new_latents)
    return new_latents
end

"""
iterate over event concepts and evaluate predicates to active/deactive
"""
@gen function event_kernel(state::CPState, event_concepts::Vector{Type{EventRelation}})
    # filter out invalid event relations (e.g., a collision is already active between a and b)
    #relations = valid_relations(state, event_concepts)

    # activate new events
    # map active events to weights 3D tensor for birth decision
    # evaluate predicates object-pairwise 
    weights = #call predicates event_concepts for each 
    event_idx = @trace(categorical(weights), :event_idx)

    # Switch combinator to evaluate clauses for each event, currently only one
    new_latents = @trace(Gen.Switch(map(clause, event_concepts))(event_idx, state.bullet_state), :event)
    # alternative: update latents with Map
    update_latents(state, new_latents)

    # some new events kill old events

    return active_events
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
@gen (static) function kernel(t::Int, prev_state::CPState, params::CPParams)
    # event kernel
    event_state = @trace(event_kernel(prev_state, params.event_concepts), :events)

    # simulate physics for the next step based on the current state and observe positions
    bullet_state::BulletState = PhySMC.step(params.sim, state.bullet_state)
    obs = @trace(Gen.Map(observe_position)(bullet_state.kinematics, noises), :observe)

    next_state = CPState(bullet_state, event_state)
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
