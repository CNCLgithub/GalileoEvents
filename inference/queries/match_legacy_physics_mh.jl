"""
Attempts to infer physical parameters in the Exp2 physics engine
that match observations generated from Exp1 physics engine
"""

using Gen
using CSV
using JSON
using Gadfly
using PyCall
using Gen_Compose

include("../dist.jl")

np = pyimport("numpy")
gm = pyimport("galileo_ramp.world.simulation.exp2_physics");

struct Scene
    data::Dict
    n_frames::Int
    simulator::T where T<:Function
    obs_noise::Float64
end;

@gen function generative_model(prior,
                               scene::Scene,
                               addr)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    # data["gravity"] = @trace(Gen_Compose.draw(prior, :gravity))
    ramp_obj_fric = @trace(Gen_Compose.draw(prior, :friction_a))
    table_obj_fric = @trace(Gen_Compose.draw(prior, :friction_b))
    data["objects"]["A"]["friction"] = ramp_obj_fric
    data["objects"]["B"]["friction"] = table_obj_fric

    new_state = scene.simulator(data, ["A", "B"], scene.n_frames)
    pos = new_state[1]
    @trace(mat_noise(pos, scene.obs_noise), addr)
    return nothing
end


latents = [:friction_a, :friction_b]
friction_rv = StaticDistribution{Float64}(uniform, (0.001, 0.999))
prior = Gen_Compose.DeferredPrior(latents,
                                  [friction_rv, friction_rv])

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

function run_inference(scene_data, positions, simulator, out_path)

    t = first(size(positions))
    scene = Scene(scene_data, t, simulator, 0.2)

    observations = Gen.choicemap()
    set_value!(observations, :pos, positions)
    query = StaticQuery(latents,
                        prior,
                        generative_model,
                        (scene,),
                        observations)
    procedure = MetropolisHastings()

    @time results = static_monte_carlo(procedure, query, 1000)
    plot = viz(results)
    plot |> SVG("$(out_path)_trace.svg",30cm, 30cm)
    df = to_frame(results)
    CSV.write("$(out_path)_trace.csv", df)
    return results
end

function main()
    # sim = gm.run_mc_trace
    sim = gm.run_full_trace
    scene_json = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_10.json"
    scene_pos = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_10_pos.npy"
    scene_data = Dict()
    open(scene_json, "r") do f
        # dicttxt = readstring(f)
        scene_data=JSON.parse(f)["scene"]
    end
    pos = np.load(scene_pos)
    pos = permutedims(pos, [2, 1, 3])
    run_inference(scene_data, pos, sim, "test")
    return nothing
end

main()
