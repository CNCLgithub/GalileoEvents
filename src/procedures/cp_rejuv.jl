
export cp_rejuv,
    extract_cp


""" Defines a series of reversable jump proposal for changepoints"""

"""
Proposes a random walk for incongruent density.

Only applies to traces with a detected change point
"""
@gen function incongruent_proposal(tr::Gen.Trace, cp::Int)
    choices = get_choices(tr)
    addr = :chain => cp => :physics => 1 => :persistence => :density
    if has_value(choices, addr)
        prev_dens = choices[addr]
        {addr} ~ log_uniform(prev_dens*0.9, prev_dens * 1.1)
    end
    return nothing
end

function softmax(x::Vector{Float64})
    max_x = maximum(x)
    num =  exp.(x .- max_x)
    denum = sum(exp.(x .- max_x))
    num./denum
end

function extract_cp(tr::Gen.Trace)
    t, _ = get_args(tr)
    choices = get_choices(tr)
    cp = t + 1
    for i = 1:t
        addr = :chain => i => :graph => :changepoint
        if choices[addr]
            cp = i
            break
        end
    end
    return cp
end

"""
Proposes changepoints

There are three jumps:

1. there is no cp -> propose a cp in [1,t]
2. there is a cp in [1,t] -> another cp in [1,t]
3. there is a cp in [1,t] -> a cp > t

"""

function cp_stats(state::Gen.ParticleFilterState)
    traces = Gen.get_traces(state)
    n = length(traces)
    # println(get_log_weights(state))
    # bad = findall(isinf.(Gen.get_log_weights(state)))
    # for i in bad
    #     display(get_choices(state.traces[i]))
    # end
    uw_traces = Gen.sample_unweighted_traces(state, n)
    cps = map(extract_cp, uw_traces)
    t, _ = get_args(state.traces[1])
    sums = zeros(t + 1)
    for i = 1:n
        idx = cps[i]
        sums[idx] += 1
    end
    mus = sums ./ n
end


function cp_rejuv(proc::PopParticleFilter,
                  state::Gen.ParticleFilterState)
    n = length(state.traces)
    p_cols = cp_stats(state)
    t = length(p_cols) - 1
    # ensure that at least some particles detect collision on [1,t]
    a = max(1, t-10)
    # println(p_cols[a:t])
    rejuv_cp = maximum(p_cols[a:t]) >= 0.11
    if rejuv_cp & (t > 1)
        # println("rejuv cp @ t $t")
        for i = 1:n
            trace = state.traces[i]
            cp = extract_cp(trace)
            if cp < t+1
                addr = :chain => cp => :physics => 1 => :persistence => :congruent
                (trace,_) = mh(trace, Gen.select(addr))
                (trace,_) = mh(trace, incongruent_proposal, (cp,))
            end
            state.traces[i] = trace
        end
    end

    return nothing

end
