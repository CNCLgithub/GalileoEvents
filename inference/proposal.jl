using Gen

include("dist.jl")

@gen function perturb_mass(prev_trace, params)
    choices = get_choices(prev_trace)
    mass = get_value(choices, :unknown_mass)
    @trace(trunc_norm(mass, params.mass_prior[2:end]...),
           :unknown_mass)
    return nothing
end;

mh_move(trace, prop, options) = metropolis_hastings(trace, prop, (options,))

"""
Returns a function that folds a trace over a collection moves given
a trace and options for those perturbation functions
"""
function mh_rejuvinate(moves::Array)
    return (trace, options) -> foldl(( t, p ) -> first(mh_move(t, p, options)),
                                     moves, init = trace)
end;

"""
Simple case of only perturbing mass
"""
function simple_rejuv()
    mh_rejuvinate([perturb_mass])
end
