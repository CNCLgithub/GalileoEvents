using Gen

include("dist.jl")

@gen function propose_to_addr(trace, addr, rv)
    choices = get_choices(trace)
    value = get_value(choices, addr)
    println(typeof(rv))
    rv(value, addr)
    return nothing
end;


# Applies a truncated normal perturbation at the given address
@gen function trunc_norm_perturb(prev_trace, addr, params)
    choices = get_choices(prev_trace)
    value = get_value(choices, addr)
    @trace(trunc_norm(value, params...), addr)
    return nothing
end;

# mh_move(trace, prop, options) = metropolis_hastings(trace, prop, (options,))

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
function gen_seq_trunc_norm(latents::Array, rv_params::Array)
    n_latents = length(latents)
    blocks = mh_rejuvenate(repeat([trunc_norm_perturb], n_latents))
    return trace -> blocks(trace, zip(latents, rv_params))
end;

"""
Returns a function that applies a batch of proposals
to a given trace.
"""
function propose_batch(addresses, rvs)
    @gen function f(trace)
        # map((addr, rv) -> propose_to_addr(trace, addr, rv),
        #     zip(addresses, rvs))
        for (addr, rv) in zip(addresses, rvs)
            propose_to_addr(trace, addr, rv)
        end
        return Nothing
    end;
    return f
end;

function gen_all_prop(scene::Scene, prop_params)
    addresses = []
    rvs = [(mu,a) -> @trace(trunc_norm(mu, ps...), a) for ps in prop_params]
    for ball in keys(scene.balls)
        for l in scene.latents
            push!(addresses, ball => l)
        end
    end
    prop = propose_batch(addresses, repeat(rvs, length(scene.balls)))
    return (t -> first(mh(t, prop, tuple())), addresses)

end;

function gen_stupid_proposal(scene, prop::Matrix{Float64})
    addresses = []
    for ball in keys(scene.balls)
        for l in scene.latents
            push!(addresses, ball => l)
        end
    end
    @gen function f(trace)
        choices = get_choices(trace)
        for ball in keys(scene.balls)
            for l in scene.latents
                v = get_value(choices, ball => l)
                @trace(uniform(v - 0.1, v + 0.1), ball => l)
                # @trace(trunc_norm(v, prop...), ball => l)
            end
        end
    end
    return (t -> first(mh(t, f, tuple()))), addresses
end;

function gen_gibbs_proposal(scene, prop::Matrix{Float64})
    addresses = []
    for ball in keys(scene.balls)
        for l in scene.latents
            push!(addresses, ball => l)
        end
    end
    return gen_seq_trunc_norm(addresses, prop), addresses
end;
