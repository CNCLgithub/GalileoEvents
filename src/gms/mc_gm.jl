export MCParams,
    MCState,
    mc_gm

################################################################################
# Generative Model
################################################################################
"""
Parameters for Markov-simulation model

$(TYPEDEF)

---

$(TYPEDFIELDS)
"""
struct MCParams <: GMParams
    # prior
    material_prior::MaterialPrior
    physics_prior::PhysPrior
    # simulation
    sim::BulletSim
    template::BulletState
    n_objects::Int64
    obs_noise::Float64
end

"""
$(TYPEDSIGNATURES)

Initializes `MCParams` from a constructed scene in pybullet.
"""
function MCParams(client::Int64, objs::Vector{Int64},
                  mprior::MaterialPrior, pprior::PhysPrior,
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

    MCParams(mprior, pprior, sim, template, length(objs), obs_noise)
end

struct MCState <: GMState
    bullet_state::BulletState
end


################################################################################
# Gen code
################################################################################


@gen function mc_object_prior(ls::RigidBodyLatents, gm::MCParams)
    # sample material
    mi = @trace(categorical(gm.material_prior.material_weights), :material)
    # sample physical properties
    phys_params = gm.physics_prior
    mass_mu, mass_sd = phys_params.mass
    mass = @trace(trunc_norm(mass_mu, mass_sd, 0., Inf),
                     :mass)
    fric_mu, fric_sd = phys_params.friction
    friction = @trace(trunc_norm(fric_mu,fric_sd, 0., 1.),
                      :friction)
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

@gen function mc_prior(gm::MCParams)
    latents = gm.template.latents
    gms = Fill(gm, length(latents))
    new_latents = @trace(Gen.Map(mc_object_prior)(latents, gms), :objects)
    bullet_state = setproperties(gm.template; latents = new_latents)
    init_state = MCState(bullet_state)
    return init_state
end

@gen function observe_position(k::RigidBodyState, noise::Float64)
    pos = k.position # XYZ position
    # add noise to position
    obs = @trace(broadcasted_normal(pos, noise), :position)
    return obs
end

@gen function kernel(t::Int, prev_state::MCState, gm::MCParams)
    sim_step::BulletState = PhySMC.step(gm.sim, prev_state.bullet_state)
    noises = Fill(gm.obs_noise, length(sim_step.kinematics))
    obs = @trace(Gen.Map(observe_position)(sim_step.kinematics, noises), :observe)
    next_state = MCState(sim_step)
    return next_state
end

@gen function mc_gm(t::Int, gm::MCParams)
    init_state = @trace(mc_prior(gm), :prior)
    # simulate `t` timesteps
    println(init_state)
    states = @trace(Gen.Unfold(kernel)(t, init_state, gm), :kernel)
    return states
end
