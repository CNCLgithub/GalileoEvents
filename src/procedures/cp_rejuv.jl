
export cp_rejuv


""" Defines a series of reversable jump proposal for changepoints"""


# TODO have more effecient cp lookup
"""
Proposes a random walk for incongruent density.

Only applies to traces with a detected change point
"""
@gen function incongruent_proposal(tr::Gen.Trace, cp::Int)
    t, _ = get_args(tr)
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
    cp = 0
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

Only well defined for c < t and c == t
"""
@gen function cp_proposal(tr::Gen.Trace, ps::Vector{Float64})
    t, _ = get_args(tr)
    rets = get_retval(tr)
    _, prev_graph, _ = rets[end-1]
    state,graph,belief = last(rets)
    col1 = first(prev_graph)
    col2 = first(graph)
    weights = zeros(t)
    old_cp = extract_cp(tr)
    # println("prev cp: $(old_cp) @ t $(t)")
    if old_cp == 0
        # cp > t -> cp <= t
        weights = softmax(ps)
    elseif old_cp == t
        # (cp == t) -> (cp < t)
        weights[1:(t-1)] = softmax(ps[1:(t-1)])
    else
        # (cp < t) -> (cp == t)
        weights[(old_cp+1):t] = softmax(ps[(old_cp+1):t])
    end
    new_cp = ({:cp} ~ categorical(weights))
    return (old_cp, new_cp)
end

@involution function cp_involution(model_args, proposal_args, proposal_retval)
    old_cp, new_cp = proposal_retval
    # println("$(old_cp) -> $(new_cp)")
    addr = :chain => new_cp => :graph => :collision
    @write_discrete_to_model(addr, true)
    if (old_cp < new_cp) & old_cp > 0
        for i = old_cp:(new_cp - 1)
            addr = :chain => i => :graph => :collision
            @write_discrete_to_model(addr, false)
        end
    end
    @write_discrete_to_proposal(:cp, old_cp)
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
    if rejuv_cp
        println("rejuv cp @ t $t")
        for i = 1:n
            (state.traces[i],_) = mh(state.traces[i], cp_proposal,
                                     (p_cols,), cp_involution)
            cp = extract_cp(state.traces[i])
            if cp > 0
                (state.traces[i],_) = mh(state.traces[i], incongruent_proposal, (cp,))
            end
        end
    end

    return nothing

end
