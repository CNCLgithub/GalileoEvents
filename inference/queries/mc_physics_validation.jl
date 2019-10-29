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

@gen function generative_model(prior::DeferredPrior, scene::Scene, addr)
    # Add the features from the latents to the scene descriptions
    data = Dict()
    data["friction"] = @trace(Gen_Compose.draw(prior, :friction))
    data["mass"] = @trace(Gen_Compose.draw(prior, :mass))

    state = scene.simulator(data = data, T = scene.n_frames, pad = 0)
    # state = scene.simulator(scene.n_frames, data)
    pos = state[1]
    @trace(mat_noise(pos, scene.obs_noise), addr)
end


latents = [:mass, :friction]
prior = Gen_Compose.DeferredPrior(latents,
                                  [StaticDistribution{Float64}(uniform, (0.1, 10)),
                                   StaticDistribution{Float64}(uniform, (0.001, 1.0))])

function estimate_layer(estimate)
    layer(x = :iter, y = estimate, Gadfly.Geom.beeswarm,
          Gadfly.Theme(background_color = "white"))
end

function viz(results::Gen_Compose.StaticTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(x -> Gadfly.plot(df, estimate_layer(x)),
                    results.latents)
    # last log scores
    log_scores = Gadfly.plot(df, estimate_layer(:log_score))
    plot = vstack(estimates..., log_scores)
end

function run_inference(positions, simulator, out_path, iterations = 100)
    scene = Scene(Dict(), size(positions)[1], simulator, 0.2)
    observations = Gen.choicemap()
    set_value!(observations, :pos, positions)
    query = StaticQuery(latents,
                        prior,
                        generative_model,
                        (scene,),
                        observations)
    procedure = MetropolisHastings()
    results = static_monte_carlo(procedure, query, iterations)
    plot = viz(results)
    plot |> SVG("test_trace.svg",30cm, 30cm)
    df = to_frame(results)
    CSV.write("test_trace.csv", df)
    return results
end

function main()
    sim = gm.run_mc_trace
    # sim = gm.run_full_trace
    t = 10
    obj_data = Dict([("friction", 0.2)])
    obs = first(gm.run_full_trace(t, obj_data))
    run_inference(obs, sim, "test", 100)
    return nothing
end

main();
