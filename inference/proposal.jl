using Gen

include("dist.jl")



# Applies a truncated normal perturbation at the given address
@gen function trunc_norm_perturb(prev_trace, addr, params)
    choices = get_choices(prev_trace)
    value = get_value(choices, addr)
    @trace(trunc_norm(value, params...), addr)
    return nothing
end;

# mh_move(trace, prop, options) = metropolis_hastings(trace, prop, (options,))
function mh_move(trace, prop, options)
    println(prop)
    println(options)
    return metropolis_hastings(trace, prop, (options,))
end

"""
Returns a function that folds a trace over a collection moves given
a trace and parameters for those perturbation functions
"""
function mh_rejuvenate(moves::Array)
    return (trace, params) -> foldl((t, etc) -> first(mh(t, etc...)),
                                     zip(moves, params), init = trace)
end;

"""
Perturb each latent sequentially
using `trunc_norm_perturb`
"""
function gen_seq_trunc_norm(latents::Array{Symbol}, rv_params::Array)
    n_latents = length(latents)
    blocks = mh_rejuvenate(repeat([trunc_norm_perturb], n_latents))
    return trace -> blocks(trace, zip(latents, rv_params))
end
