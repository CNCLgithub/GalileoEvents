using Gen
using Printf

include("model.jl")
include("proposal.jl")

struct InferenceTrial
    scene::Scene
    gt
    latents::Array
end;

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

function report_step(state, latents)
    weights = get_log_weights(state)
    estimates = Matrix{Float64}(undef, length(weights), length(latents))
    for l = 1:length(latents)
        estimates[:, l] = [state.traces[i][latents[l]]
                           for i = 1:length(state.traces)]
    end
    return estimates, weights
end;

function report_step!(results::InferenceResults, state, latents, t)
    results.weights[t, :] = Gen.get_log_weights(state)
    for l = 1:length(latents)
        results.estimates[t, :, l] = [state.traces[i][latents[l]]
                                      for i = 1:length(state.traces)]
    end
    for i = 1:length(state.traces)
        pos, _, _ = get_retval(state.traces[i])

    end
end;

function particle_filter(trial::InferenceTrial,
                         params::InferenceParams)
    # construct initial observations
    model = make_model(trial.scene)
    ess = params.n_particles * params.resample
    obs, frames = make_obs(model, trial.gt, steps = params.steps)
    results = InferenceResults(
        Matrix{Float64}(undef, length(frames), params.n_particles),
        Array{Float64,3}(undef, length(frames),
                         params.n_particles,
                         length(trial.latents)))
    init_obs = choicemap()
    set_submap!(init_obs, :obs => frames[1],
                get_submap(obs, :obs => frames[1]))

    state = Gen.initialize_particle_filter(model, (frames[1],), init_obs,
                                           params.n_particles)
    report_step!(results, state, trial.latents, 1)
    for (it, t) in enumerate(frames[2:end])
        # apply a rejuvenation move to each particle
        for p=1:params.n_particles
            state.traces[p] = params.rejuv(state.traces[p], trial.scene)
        end
        # Resample depending on ess
        Gen.maybe_resample!(state, ess_threshold=ess, verbose = true)
        # Step to next observation
        next_obs = choicemap()
        set_submap!(next_obs, :obs => t, get_submap(obs, :obs => t))
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), next_obs)
        # Report step
        @printf "Iteration %d / %d\n" (it+1) length(frames)
        report_step!(results, state, trial.latents, it + 1)
    end
    return results, frames
end;

"""
Function exposed to python interface
"""
function run_inference(scene_args, inf_args, perturb::Int)

    tower, unknown, mass_prior, gt, latents = scene_args
    # (tower json, unknown block ids, mass prior)
    scene = Scene(tower, unknown, mass_prior)
    trial = InferenceTrial(scene, gt, map(Symbol, latents))

    if perturb > 0
        rejuv = simple_rejuv()
    else
        rejuv = (x,y...) -> x
    end
    params = InferenceParams(inf_args..., rejuv)
    @time results, frames = particle_filter(trial, params)
    d = Dict([("estimates", results.estimates),
              ("scores", results.weights),
              ("xs", collect(frames)),])
    return d
end;
