using Gen
using JSON
using Printf
using Base.Iterators
using ProgressMeter

include("model.jl")
include("proposal.jl")


struct InferenceParams
    steps::Int
    factorize::Bool
    rejuv::Function
    prop::Matrix{Float64}
end;

struct InferenceResults
    weights::Matrix{Float64}
    estimates::Array{Float64, 3}
end

struct MHBlock
    model
    args::Tuple{Int, Array{String}}
    obs
    prop::Function
    addrs
    size::Tuple
end;

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
end;

"""
Records inference results to two arrays.
"""
function report_step!(weights, estimates, state, latents, p)
    weights[p] = Gen.get_score(state)
    choices = Gen.get_choices(state)
    for l = 1:length(latents)
        if !Gen.has_value(choices, latents[l])
            continue
        end
        estimates[p, l] = choices[latents[l]]
    end
end;

"""
Peforms MH-MCMC on a given observation
"""
function resim_mh!(params::MHBlock)

    steps = params.size[1]
    weights = Array{Float64, 1}(undef, steps)
    estimates = fill(NaN, params.size)

    # Gen's `generate` function accepts a model, a tuple of arguments to the model,
    # and a ChoiceMap representing observations (or constraints to satisfy). It returns
    # a complete trace consistent with the observations, and an importance weight.
    (tr, _) = generate(params.model, params.args, params.obs)

    # Perform resimulation updates
    @showprogress 1 "Running resim-mh..." for it=1:steps
        tr = params.prop(tr)
        report_step!(weights, estimates, tr,
                     params.addrs, it)
    end
    return weights, estimates
end;


"""
Performs SMC using MH-MCMC across several observations
"""
function smc(trial::Scene, params::InferenceParams)
    # construct initial observations
    model = make_model(trial)
    obs, args = make_obs(trial, factorize = params.factorize)
    result_size = (length(args), params.steps,
                   length(trial.balls) * length(trial.latents))
    results = InferenceResults(
        Matrix{Float64}(undef, length(args), params.steps),
        fill(NaN, result_size))

    for (it, (t, active)) in enumerate(args)
        # Step to next observation
        cur_obs = choicemap()
        set_submap!(cur_obs, :obs => t, get_submap(obs, :obs => t))

        # Define the proposal function
        addrs = [o => l for l in trial.latents
                 for o in filter(x-> x!="", active)]
        n_active = length(filter(x-> x!="", active))
        prop_params = collect(eachrow(repeat(params.prop, n_active)))
        rejuv = params.rejuv(addrs, prop_params)
        block = MHBlock(model, (t, active), cur_obs, rejuv,
                        addrs, result_size[2:end])
        # Apply the proposal function to the MH procedure
        results.weights[it, :], results.estimates[it, :, :] = resim_mh!(block)
    end
    return results, args
end;

"""
Function exposed to python interface
"""
function run_inference(scene_args, dist_args, inf_args)

    gt, balls, latents, ts = scene_args
    # (tower json, unknown block ids, mass prior)
    scene = Scene(scene_args..., dist_args["prior"])
    rejuv = gen_gibbs_trunc_norm
    params = InferenceParams(inf_args..., rejuv, dist_args["prop"])
    @time results, args = smc(scene, params)
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
# struct InferenceParams
#     steps::Int
#     factorize::Bool
#     rejuv::Function
#     prop::Matrix{Float64}
# end;

function test_inf(trial_path)

    str = String(read(trial_path))
    dict = JSON.parse(str)["scene"]

    balls = collect(keys(dict["objects"]))
    ts = [10, 100]
    scene_args = (dict, balls, ["density"], ts)

    dist_args = Dict("prior" => [[-2 2];],
                     "prop" => [[0.1 -2 2];])
    inf_args = [2000, true]
    d = run_inference(scene_args, dist_args, inf_args)
    print(json(d, 4))
end;
