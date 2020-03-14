"""
Runs inference over the `test_physics` scene
"""

using Gen
using CSV
using Gadfly
using PyCall
using Gen_Compose

include("../dist.jl")

gm = pyimport("galileo_ramp.world.simulation.test_physics");

struct Scene
    data::Dict
    n_frames::Int
    simulator::T where T<:Function
    obs_noise::Float64
end;

@gen function generative_model(state,
                               t::Int,
                               scene::Scene,
                               prior::DeferredPrior,
                               addr)
    # Add the features from the latents to the scene descriptions
    data = Dict()
    data["friction"] = @trace(Gen_Compose.draw(prior, :friction))
    data["mass"] = @trace(Gen_Compose.draw(prior, :mass))

    new_state = scene.simulator(data = data, state = state, pad = 0)
    # new_state = scene.simulator(t, data)
    pos = new_state[1][end, :]
    @trace(random_vec(pos, scene.obs_noise), addr)
    return new_state
end


latents = [:mass, :friction]
prior = Gen_Compose.DeferredPrior(latents,
                                  [StaticDistribution{Float64}(uniform, (0.1, 10)),
                                   StaticDistribution{Float64}(uniform, (0.001, 0.999))])
moves = [DynamicDistribution{Float64}(uniform, x -> (x-0.1, x+0.1))
         DynamicDistribution{Float64}(uniform, x -> (x-0.01, x+0.01))]

# the rejuvination will follow Gibbs sampling
rejuv = gibbs_steps(moves, latents)

function estimate_layer(estimate)
    layer(x = :t, y = estimate, Gadfly.Geom.beeswarm,
          Gadfly.Theme(background_color = "white"))
end

function viz(results::Gen_Compose.SequentialTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(x -> Gadfly.plot(df, estimate_layer(x)),
                    results.latents)
    # last log scores
    log_scores = Gadfly.plot(df, estimate_layer(:log_score))
    plot = vstack(estimates..., log_scores)
end

function run_inference(positions, simulator, out_path)
    scene = Scene(Dict(), size(positions)[1], simulator, 0.2)
    observations = Gen.choicemap()
    set_value!(observations, :pos, positions)
    query = SequentialQuery(latents,
                        prior,
                        generative_model,
                        (scene,),
                        observations)
    # -----------------------------------------------------------
    # Define the inference procedure
    # In this case we will be using a particle filter
    #
    # Additionally, this will be under the Sequential Monte-Carlo
    # paradigm.
    n_particles = 10
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    results = sequential_monte_carlo(procedure, query)
    plot = viz(results)
    plot |> SVG("test_trace.svg",30cm, 30cm)
    df = to_frame(results)
    println(df)
    CSV.write("test_trace.csv", df)
    return results
end

function main()
    sim = gm.run_mc_trace
    # sim = gm.run_full_trace
    t = 10
    obj_data = Dict([("friction", 0.2)])
    obs_t = first(gm.run_full_trace(t, obj_data))
    obs = Vector{Vector{Float64}}(undef, t)
    for i = 1:t
        obs[i] = obs_t[i, :]
    end
    run_inference(obs, sim, "test")
    return nothing
end

main();
