using Gen
using JSON
using Printf
using Base.Iterators

include("model.jl")
include("proposal.jl")


struct InferenceParams
    n_particles::Int
    steps::Int
    resample::Real
    factor::Bool
    rejuv::Function
    prop::Matrix{Float64}
end;

struct InferenceResults
    weights::Matrix{Float64}
    estimates::Array{Float64, 3}
end

## Particle filter

function JSON.lower(c::ChoiceMap)
    d = Dict(Gen.get_values_shallow(c))
    others = Gen.get_submaps_shallow(c)
    if isempty(others)
        return d
    end
    other_keys, other_maps = zip(others...)
    other_vals = map(JSON.lower, other_maps)
    others = Dict(zip(other_keys, other_vals))
    return merge(d, others)
end

function report_step!(results::InferenceResults, state, latents, t)
    results.weights[t, :] = Gen.get_log_weights(state)
    for p = 1:length(state.traces)
        choices = Gen.get_choices(state.traces[p])
        for l = 1:length(latents)
            if !Gen.has_value(choices, latents[l])
                continue
            end
            results.estimates[t, p, l] = choices[latents[l]]
        end
    end
end;

function particle_filter(trial::Scene,
                         params::InferenceParams)
    # construct initial observations
    model = make_model(trial)
    ess = params.n_particles * params.resample
    obs, args = make_obs(trial, steps = params.steps, factor = params.factor)
    results = InferenceResults(
        Matrix{Float64}(undef, length(args), params.n_particles),
        fill(NaN, (length(args), params.n_particles,
                   length(trial.balls) * length(trial.latents))))

    let
        state = Nothing;
    for (it, (t, active)) in enumerate(args)
        # Step to next observation
        cur_obs = choicemap()
        set_submap!(cur_obs, :obs => t, get_submap(obs, :obs => t))

        if it == 1
            state = Gen.initialize_particle_filter(model, (t,active), cur_obs,
                                                   params.n_particles)
        else
            Gen.particle_filter_step!(state, (t,active), (UnknownChange(),),
                                      cur_obs)
        end
        println(active)
        # apply a rejuvenation move to each particle
        addrs = [o => l for l in trial.latents
                 for o in filter(x-> x!="", active)]
        n_active = length(filter(x-> x!="", active))
        prop_params = collect(eachrow(repeat(params.prop, n_active)))
        rejuv = params.rejuv(addrs, prop_params)
        for p=1:params.n_particles
            state.traces[p] = rejuv(state.traces[p])
        end
        # Resample depending on ess
        Gen.maybe_resample!(state, ess_threshold=ess, verbose = true)
        # Report step
        @printf "Iteration %d / %d\n" it params.steps
        report_step!(results, state, addrs, it)
    end
    end
    return results, args
end;

"""
Function exposed to python interface
"""
function run_inference(scene_args, dist_args, inf_args)

    gt, balls, latents, nf = scene_args
    # (tower json, unknown block ids, mass prior)
    scene = Scene(scene_args..., dist_args["prior"])
    # rejuv, addrs = gen_stupid_proposal(scene, dist_args["prop"])
    rejuv = gen_gibbs_trunc_norm
    params = InferenceParams(inf_args..., rejuv, dist_args["prop"])
    @time results, args = particle_filter(scene, params)
    frames = map(first, args)
    addrs = [o => l for l in latents for o in balls]
    d = Dict([("estimates", results.estimates),
              ("scores", results.weights),
              ("xs", collect(frames)),
              ("latents", addrs),
              ("gt", [gt["objects"][o][l] for l in latents
                 for o in balls]),])
    return d
end;

function test_inf(trial_path)

    str = String(read(trial_path))
    dict = JSON.parse(str)["scene"]

    balls = collect(keys(dict["objects"]))
    scene_args = (dict, balls, ["density"], 900)
    dist_args = Dict("prior" => [[-2 2];],
                     "prop" => [[0.1 -2 2];])
    inf_args = [10, 9, 0.5, false]
    d = run_inference(scene_args, dist_args, inf_args)
    print(json(d, 4))
end;
