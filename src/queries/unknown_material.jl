"""
Simple example of inferring density and friction with no visual prior
"""

using Gen
using Gen_Compose

include("../distributions.jl")
include("../gms/mc_gm.jl")

Gen.load_generated_functions()

@gen function gibbs_step(trace)
    (t, p) = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    for i = 1:p.n_objects
        density = choices[:object_physics => i => :density]
        friction = choices[:object_physics => i => :friction]
        @trace(log_uniform(density, 0.1), :object_physics => i => :density)
        @trace(log_uniform(friction, 0.1), :object_physics => i => :friction)
    end
end

function rejuv(trace)
    (new_trace, _) = Gen.mh(trace, gibbs_step, tuple())
    return new_trace
end

function run_inference(args, init_obs, init_args,
                       observations, iter::Int = 100)

    latents = Dict( :x => x -> :x )
    query = Gen_Compose.SequentialQuery(latents, #bogus for now
                                        generative_model,
                                        init_args,
                                        init_obs,
                                        args,
                                        observations)


    # -----------------------------------------------------------
    # Define the inference procedure
    # In this case we will be using a particle filter
    #
    # Additionally, this will be under the Sequential Monte-Carlo
    # paradigm.
    n_particles = iter
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    @time sequential_monte_carlo(procedure, query)
    # @profview sequential_monte_carlo(procedure, query)
end

function main()

    n = 20

    cm = choicemap()
    cm[:initial_state => 1 => :init_pos] = 1.5
    cm[:initial_state => 2 => :init_pos] = 0.5
    cm[:object_physics => 1 => :density] = 6.0
    cm[:object_physics => 2 => :density] = 6.0
    cm[:object_physics => 1 => :friction] = 0.2
    cm[:object_physics => 2 => :friction] = 0.2
    trace, w = Gen.generate(generative_model, (n, default_params), cm)
    choices = Gen.get_choices(trace)
    init_obs = choicemap()
    set_submap!(init_obs, :initial_state, get_submap(cm, :initial_state))

    obs = Vector{Gen.ChoiceMap}(undef, n)
    for t = 1:n
        tcm = choicemap()
        addr = :chain => t => :positions
        tcm[addr] = choices[addr]
        obs[t] = tcm
    end

    args = [(t, default_params) for t in 1:n]

    results = run_inference(args, init_obs, (0, default_params), obs, 20)

end
