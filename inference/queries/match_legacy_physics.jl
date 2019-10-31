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

@gen function generative_model(state,
                               t::Int,
                               scene::Scene,
                               prior::DeferredPrior,
                               addr)
    # Add the features from the latents to the scene descriptions
    data = deepcopy(scene.data)
    data["gravity"] = @trace(Gen_Compose.draw(prior, :gravity))
    ramp_obj_fric = @trace(Gen_Compose.draw(prior, :friction_a))
    table_obj_fric = @trace(Gen_Compose.draw(prior, :friction_b))
    data["objects"]["A"]["friction"] = ramp_obj_fric
    data["objects"]["B"]["friction"] = table_obj_fric

    new_state = scene.simulator(data, ["A", "B"],
                                state = state, pad = 0)
    pos = new_state[1]
    @trace(mat_noise(pos, scene.obs_noise), addr)
    return new_state
end


latents = [:gravity, :friction_a, :friction_b]
friction_rv = StaticDistribution{Float64}(uniform, (0.001, 0.999))
prior = Gen_Compose.DeferredPrior(latents,
                                  [StaticDistribution{Float64}(uniform, (-20, -0.1)),
                                   friction_rv,
                                   friction_rv])
moves = [DynamicDistribution{Float64}(uniform, x -> (x-0.1, x+0.1))
         DynamicDistribution{Float64}(uniform, x -> (x-0.01, x+0.01))
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

function run_inference(scene_data, positions, simulator, out_path)

    t = first(size(positions))
    scene = Scene(scene_data, t, simulator, 0.2)

    obs = Vector{Array{Float64, 2}}(undef, t)
    for i = 1:t
        obs[i] = positions[i, :, :]
    end

    observations = Gen.choicemap()
    set_value!(observations, :pos, obs)
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
    n_particles = 100
    ess = n_particles * 0.5
    # defines the random variables used in rejuvination
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
    scene_json = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_0.json"
    scene_pos = "/home/mario/dev/data/galileo-ramp/scenes/legacy_converted/trial_0_pos.npy"
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
