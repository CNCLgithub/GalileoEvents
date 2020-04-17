export attention

using Statistics
using PhysicalConstants

function entropy(ps::AbstractArray{Float64})
    -k_B * sum(map(p -> p * log(p)))
end

function lookback() end

function lookforward(traces,
                     query::StaticQuery)

    n = length(traces)
    upfn = t,o -> update(t, query.args, (UnknownChange(),), o)

    indeces = multinomial(fill(1.0/n, Int(n/2.0)))
   

    new_traces = Vector{Gen.Trace}(undef, n)
    weights = Vector{Float64}(undef, n)
    for i = 1:n
        new_traces[i], weights[i],_,_ = upfn(traces[i])
    end
    estimates = map(t -> parse_trace(query, t), new_traces)
    return (new_traces, weights, estimates[:ramp_density])

end

function attention(state::Gen.ParticleFilterState,
                   query::SequentialQuery;
                   forward::Int = 2,
                   backward::Int = 2)

    n = length(state.traces)
    t, params = Gen.get_args(first(state.traces))
    traces = Gen.sample_unweighted_traces(state, n)
    vars = Vector{Float64}(undef, forward)
    s = Vector{Float64}(undef, forward)

    # forward step
    for i = t+1:t+forward
        target = query[i]
        traces, weights, estimates = lookforward(traces, target)
        vars[i] = var(estimates)
        s[i] = entropy(weights)
    end

    # find the OLS derivative for entropy
    ts = collect(1:forward)
    b = (ts'*ts)\ts'*s
    println("b $(b)")
    println(s)
    return abs(b)
end
