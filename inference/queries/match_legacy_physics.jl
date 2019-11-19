"""
Attempts to infer physical parameters in the Exp2 physics engine
that match observations generated from Exp1 physics engine
"""

using Gen
using CSV
using JSON
using Gadfly
using PyCall
using Compose
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

@gen function generative_model(state,
                               t::Int,
                               scene::Scene,
                               prior::DeferredPrior,
                               addr)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    data["objects"]["A"]["mass"] = @trace(Gen_Compose.draw(prior, :mass_a))
    data["objects"]["B"]["mass"] = @trace(Gen_Compose.draw(prior, :mass_b))
    data["objects"]["A"]["friction"] = @trace(Gen_Compose.draw(prior, :friction_a))
    data["objects"]["B"]["friction"] = @trace(Gen_Compose.draw(prior, :friction_b))
    ground_fric = @trace(Gen_Compose.draw(prior, :friction_ground))
    data["ramp"]["friction"] = ground_fric
    data["table"]["friction"] = ground_fric

    new_state = scene.simulator(data, ["A", "B"], state = state)
    pos = new_state[1]
    @trace(mat_noise(pos, scene.obs_noise), addr)
    return new_state
end

function estimate_layer(estimate, with_map = false)
    l = layer(x = :t, y = estimate, Gadfly.Geom.histogram2d)
    return l
end

function viz(results::Gen_Compose.SequentialTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(x -> Gadfly.plot(df, estimate_layer(x)),
                    results.latents)
    # last log scores
    log_scores = Gadfly.plot(df, estimate_layer(:log_score))
    plot = vstack(estimates..., log_scores)
    compose(compose(context(), rectangle(), fill("white")),
            plot)
end

function run_inference(scene_data, positions, out_path)

    t = first(size(positions))
    scene = Scene(scene_data, t, gm.run_mc_trace, 0.2)

    obs = Vector{Array{Float64, 2}}(undef, t)
    for i = 1:t
        obs[i] = positions[i, :, :]
    end

    observations = Gen.choicemap()
    set_value!(observations, :pos, obs)

    latents = [:mass_a, :mass_b, :friction_a, :friction_b, :friction_ground]
    mass_rv = StaticDistribution{Float64}(uniform, (0.1, 200))
    friction_rv = StaticDistribution{Float64}(uniform, (0.001, 0.999))
    prior = Gen_Compose.DeferredPrior(latents,
                                      [mass_rv,
                                       mass_rv,
                                       friction_rv,
                                       friction_rv,
                                       friction_rv])
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
    n_particles = 20
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
    mass_move = DynamicDistribution{Float64}(normal, x -> (x, 20.0))
    friction_move = DynamicDistribution{Float64}(normal, x -> (x, 0.2))
    moves = [mass_move
             mass_move
             friction_move
             friction_move
             friction_move]
    # the rejuvination will follow Gibbs sampling
    rejuv = gibbs_steps(moves, latents)
    procedure = ParticleFilter(n_particles,
                               ess,
                               rejuv)

    @time results = sequential_monte_carlo(procedure, query)
    plot = viz(results)
    plot |> SVG("$(out_path)_trace.svg",30cm, 30cm)
    df = to_frame(results)
    CSV.write("$(out_path)_trace.csv", df)
    return results
end

function main()
    sim = gm.run_mc_trace
    scene_json = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_24.json"
    scene_pos = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_24_pos.npy"
    scene_data = Dict()
    open(scene_json, "r") do f
        # dicttxt = readstring(f)
        scene_data=JSON.parse(f)["scene"]
    end
    pos = np.load(scene_pos)
    pos = permutedims(pos, [2, 1, 3])
    run_inference(scene_data, pos, "test_pf")
    return nothing
end
