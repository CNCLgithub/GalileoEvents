using Gen
using JSON
using Printf

include("model.jl")
include("proposal.jl")


struct InferenceParams
    n_particles::Int
    steps::Int
    resample::Real
    rejuv
end;

struct InferenceResults
    weights::Matrix{Float64}
    estimates::Array{Float64}
end

## Particle filter

function report_step!(results::InferenceResults, state, latents, t)
    results.weights[t, :] = Gen.get_log_weights(state)
    for l = 1:length(latents)
        results.estimates[t, :, l] = [state.traces[i][latents[l]]
                                      for i = 1:length(state.traces)]
    end
end;

function particle_filter(trial::Scene,
                         params::InferenceParams,
                         addrs::Array)
    # construct initial observations
    model = make_model(trial)
    ess = params.n_particles * params.resample
    obs, frames = make_obs(trial, steps = params.steps)
    results = InferenceResults(
        Matrix{Float64}(undef, length(frames), params.n_particles),
        Array{Float64,3}(undef, length(frames),
                         params.n_particles,
                         length(addrs)))
    init_obs = choicemap()
    set_submap!(init_obs, :obs => frames[1],
                get_submap(obs, :obs => frames[1]))

    println(init_obs)
    state = Gen.initialize_particle_filter(model, (frames[1],), init_obs,
                                           params.n_particles)
    report_step!(results, state, addrs, 1)
    for (it, t) in enumerate(frames[2:end])
        # apply a rejuvenation move to each particle
        for p=1:params.n_particles
            state.traces[p] = params.rejuv(state.traces[p])
        end
        # Resample depending on ess
        Gen.maybe_resample!(state, ess_threshold=ess, verbose = true)
        # Step to next observation
        next_obs = choicemap()
        set_submap!(next_obs, :obs => t, get_submap(obs, :obs => t))
        println(next_obs)
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), next_obs)
        # Report step
        @printf "Iteration %d / %d\n" (it+1) length(frames)
        report_step!(results, state, addrs, it + 1)
    end
    return results, frames
end;

"""
Function exposed to python interface
"""
function run_inference(scene_args, dist_args, inf_args)

    gt, balls, latents, nf = scene_args
    # (tower json, unknown block ids, mass prior)
    scene = Scene(scene_args..., dist_args["prior"])
    println(typeof(scene.prior))

    # rejuv, addrs = gen_stupid_proposal(scene, dist_args["prop"])
    rejuv, addrs = gen_gibbs_proposal(scene, dist_args["prop"])

    params = InferenceParams(inf_args[1:(end-1)]..., rejuv)
    @time results, frames = particle_filter(scene, params, addrs)
    d = Dict([("estimates", results.estimates),
              ("scores", results.weights),
              ("xs", collect(frames)),
              ("latents", addrs),
              ("gt", [scene.balls[l[1]][l[2]] for l in addrs]),])
    return d
end;

function test_inf(trial_path)

    str = String(read(trial_path))
    dict = JSON.parse(str)["scene"]


    scene_args = (dict, dict["objects"], ["density"], 900)
    dist_args = Dict("prior" => [[-0.0001 0.0001];],
                     "prop" => [[0.2 0.01 13.];],)
    inf_args = [1, 10, 0.5, 1]
    d = run_inference(scene_args, dist_args, inf_args)
    print(json(d, 4))
end;
