
export cp_rejuv


""" Defines a series of reversable jump proposal for changepoints"""


@gen function congruency_proposal(tr::Gen.Trace, cp::Int)
    choices = get_choices(tr)
    addr = :chain => cp => :physics => 1 => :persistence => :congruent
    prev_choice = choices[addr]
    return (prev_choice, addr)
end

@involution function congruency_involution(model_args, proposal_args, prop_ret)
    prev_choice, addr = prop_ret
    new_choice = !prev_choice
    @write_discrete_to_model(addr, new_choice)
end

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
        addr = :chain => i => :graph => :collision
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
@gen function cp_proposal(tr::Gen.Trace, ps::Vector{Float64})
    t, _ = get_args(tr)
    rets = get_retval(tr)
    _, prev_graph, _ = rets[end-1]
    state,graph,belief = last(rets)
    col1 = first(prev_graph)
    col2 = first(graph)
    weights = zeros(t + 1)
    weights[1:t] = ps
    weights[t+1] = prod(1.0 .- ps)
    old_cp = extract_cp(tr)
    # println("prev cp: $(old_cp) @ t $(t)")
    weights[old_cp] = 0
    weights = softmax(weights)
    new_cp = ({:cp} ~ categorical(weights))
    return (t, old_cp, new_cp)
end

@involution function cp_involution(model_args, proposal_args, proposal_retval)
    t, old_cp, new_cp = proposal_retval
    # println("$(old_cp) -> $(new_cp)")
    @write_discrete_to_proposal(:cp, old_cp)
    if new_cp <= t
        addr = :chain => new_cp => :graph => :collision
        @write_discrete_to_model(addr, true)
    end
    if old_cp < new_cp
        for i = old_cp:(new_cp - 1)
            addr = :chain => i => :graph => :collision
            @write_discrete_to_model(addr, false)
        end
    end
end

function extract_collisions(tr)
    t,_ = get_args(tr)
    choices = get_choices(tr)
    cols = Vector{Float64}(undef, t)
    for i = 1:t
        addr = :chain => i => :graph => :collision
        cols[i] = choices[addr]
    end
    reshape(cols, (1,t))
end

function cp_stats(state::Gen.ParticleFilterState)
    traces = Gen.get_traces(state)
    n = length(traces)
    uw_traces = Gen.sample_unweighted_traces(state, n)
    estimates = map(extract_collisions, uw_traces)
    estimates = vcat(estimates...)
    mus = (n == 1) ? estimates : mean(estimates, dims = 1)
    vec(mus)
end


function cp_rejuv(proc::PopParticleFilter,
                  state::Gen.ParticleFilterState)
    n = length(state.traces)
    p_cols = cp_stats(state)
    # ensure that at least some particles detect collision on [1,t]
    t, _ = get_args(state.traces[1])
    a = max(1, t-10)
    rejuv_cp = maximum(p_cols[a:t]) >= 0.5
    rejuv_cp = p_cols[t] > 0.1
    if rejuv_cp
        println("rejuv cp @ t $t")
        for i = 1:n
            trace = state.traces[i]
            (trace,_) = mh(trace, cp_proposal,
                           (p_cols,), cp_involution)
            cp = extract_cp(trace)
            if cp < t+1
                (trace,_) = mh(trace, congruency_proposal, (cp,),
                               congruency_involution)
                (trace,_) = mh(trace, incongruent_proposal, (cp,))
            end
        end
    end

    return nothing

end
