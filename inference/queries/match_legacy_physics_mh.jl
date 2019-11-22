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
import Cairo

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
    data["objects"]["A"]["density"] = @trace(Gen_Compose.draw(prior, :density_a))
    data["objects"]["A"]["mass"] = data["objects"]["A"]["density"] * data["objects"]["A"]["volume"]
    data["objects"]["B"]["density"] = @trace(Gen_Compose.draw(prior, :density_b))
    data["objects"]["B"]["mass"] = data["objects"]["B"]["density"] * data["objects"]["B"]["volume"]

    new_state = scene.simulator(data, ["A", "B"], scene.n_frames * 1.0/60.0,
                                fps = 60, time_scale = 10.0)
    pos = new_state[1]
    @trace(Gen.broadcasted_normal(pos, fill(scene.obs_noise, scene.n_frames)),
                                  addr)
    return nothing
end


function estimate_layer(estimate)
    layer(x = estimate, color = :log_score, Gadfly.Geom.histogram,
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
    compose(compose(context(), rectangle(), fill("white")),
            plot)
end


struct HMC <: Gen_Compose.InferenceProcedure
    update::T where T<:Function
    mass::Float64
    L::Int
    eps::Float64
    HMC(update, mass=0.1, L=10, eps=0.1) = new(update, mass,L,eps)
end

mutable struct HMCTrace
    current_trace::T where T<:Gen.DynamicDSLTrace
end

function Gen_Compose.initialize_procedure(proc::HMC,
                              query::StaticQuery,
                              addr)
    addr = observation_address(query)
    trace,_ = Gen.generate(query.forward_function,
                           (query.prior, query.args..., addr),
                           query.observations)
    return HMCTrace(trace)
end

function Gen_Compose.step_procedure!(state::HMCTrace,
                         proc::HMC,
                         query::StaticQuery,
                         addr,
                         step_func)
    state.current_trace = proc.update(state.current_trace)
    return nothing
end

function Gen_Compose.report_step!(results::T where T<:Gen_Compose.InferenceResult,
                      state::HMCTrace,
                      latents::Vector,
                      idx::Int)
    # copy log scores
    trace = state.current_trace
    results.log_score[idx] = Gen.get_score(trace)
    choices = Gen.get_choices(trace)
    for l = 1:length(latents)
        results.estimates[idx,1, l] = choices[latents[l]]
    end
    return nothing
end

function Gen_Compose.initialize_results(::HMC)
    (1,)
end

function Gen_Compose.mc_step!(proc::HMC,
                  state::HMCTrace,
                  selection)
    Gen.mala(state.current_trace, selection, 0.1)
    # Gen.hmc(state.current_trace, selection, proc.mass, proc.L, proc.eps)
end




function run_inference(scene_data, positions, out_path)

    t = first(size(positions))
    scene = Scene(scene_data, t, gm.run_full_trace, 0.2)

    observations = Gen.choicemap()
    set_value!(observations, :pos, positions)

    latents = [:density_a, :density_b]
    density_map = Dict()
    density_map["Wood"] = 1.0
    density_map["Brick"] = 2.0
    density_map["Iron"] = 8.0

    density_a = density_map[scene_data["objects"]["A"]["appearance"]]
    density_b = density_map[scene_data["objects"]["B"]["appearance"]]
    prior = Gen_Compose.DeferredPrior(latents,
                                      [StaticDistribution{Float64}(trunc_norm,
                                                                   (density_a, 3.0, 0., 12.)),
                                       StaticDistribution{Float64}(trunc_norm,
                                                                   (density_b, 3.0, 0., 12.))])

    query = StaticQuery(latents,
                        prior,
                        generative_model,
                        (scene,),
                        observations)
    mass_move = DynamicDistribution{Float64}(log_uniform, x -> (x, 0.1))
    moves = [mass_move
             mass_move]

    # the rejuvination will follow Gibbs sampling
    update_step = gibbs_steps(moves, latents)
    procedure = HMC(update_step)

    @time results = static_monte_carlo(procedure, query, 100)
    plot = viz(results)
    plot |> PNG("$(out_path)_trace.png",30cm, 30cm)
    df = to_frame(results)
    CSV.write("$(out_path)_trace.csv", df)
    return df
end

function main()
    scene_json = "../data/galileo-ramp/scenes/legacy_converted/trial_89.json"
    scene_pos = "../data/galileo-ramp/scenes/legacy_converted/trial_89_pos.npy"
    scene_data = Dict()
    open(scene_json, "r") do f
        # dicttxt = readstring(f)
        scene_data=JSON.parse(f)["scene"]
    end
    println(scene_data["objects"]["A"]["mass"])
    println(scene_data["objects"]["B"]["mass"])
    pos = np.load(scene_pos)
    pos = permutedims(pos, [2, 1, 3])
    run_inference(scene_data, pos, "test")
end

