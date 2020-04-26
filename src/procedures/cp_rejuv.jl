
export cp_rejuv,
    extract_cp


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
@gen function cp_proposal(tr::Gen.Trace, ps::Vector{Float64})
    t, _ = get_args(tr)
    current_cp = extract_cp(tr)
    weights = deepcopy(ps[1:end-1])
    current_weight = weights[current_cp]
    weights[current_cp] = 0
    nonzero = findall(!iszero, weights)
    n = length(nonzero)
    # the case where all particles agree
    if n == 1
        new_cp = current_cp
    else
        weights[nonzero]  = weights[nonzero] .+ (current_weight / n)
        weights = softmax(weights)
        new_cp = ({:cp} ~ categorical(weights))
    end
    println("$(current_cp) -> $(new_cp) @ t $(t) | w $(weights[new_cp])")
    return (t, current_cp, new_cp)
end

function cp_involution(trace, fwd_choices, fwd_ret,
                       proposal_args::Tuple;
                       check = false)

    choices = get_choices(trace)
    (t, current_cp, new_cp) = fwd_ret
    println("invo $current_cp -> $new_cp")

    bwd_choices = choicemap()
    bwd_choices[:cp] = current_cp

    if current_cp == new_cp
        return (trace, bwd_choices, 1.0)
    end


    constraints = choicemap()

    addr = :chain => new_cp => :graph => :changepoint
    println("$(addr) -> true")
    constraints[addr] = true

    println("swapping physics")
    set_submap!(constraints, :chain => new_cp => :physics,
                get_submap(choices, :chain => current_cp => :physics))

    set_submap!(constraints, :chain => current_cp => :physics,
                get_submap(choices, :chain => new_cp => :physics))

    addr = :chain => current_cp => :graph => :changepoint
    println("$addr -> false")
    constraints[addr] = false


    model_args = get_args(trace)
    (new_trace, weight, _, _) = update(trace, model_args, (NoChange(),), constraints)

    # println("choices")
    # display(get_submap(choices, :chain => current_cp => :physics))
    # display(get_submap(choices, :chain => new_cp => :physics))
    println("constraints")
    display(constraints)
    println("bwd choices")
    display(bwd_choices)
    println("weight $weight")
    # (bwd_score, bwd_ret) = assess(cp_proposal, (new_trace, proposal_args...), bwd_choices)
    # println("bwd score $bwd_score")
    # println(bwd_ret)
    (new_trace, bwd_choices, weight)

end

# @involution function cp_involution(model_args, proposal_args, proposal_retval)
#     (t, current_cp) = proposal_retval
#     new_cp = @read_discrete_from_proposal(:cp)
#     println("invo $current_cp -> $new_cp")
#     if new_cp <= t
#         addr = :chain => new_cp => :graph => :changepoint
#         println("$(addr) -> true")
#         @write_discrete_to_model(addr, true)

#         if current_cp <= t
#             println("swapping physics")
#             @copy_model_to_model(:chain => current_cp => :physics => 1,
#                                  :chain => new_cp => :physics => 1)
#             @copy_model_to_model(:chain => current_cp => :physics => 2,
#                                  :chain => new_cp => :physics => 2)
#         end
#     end
#     if current_cp <= t
#         addr = :chain => current_cp => :graph => :changepoint
#         println("$addr -> false")
#         @write_discrete_to_model(addr, false)
#     end
#     println(":cp -> $(current_cp)")
#     @write_discrete_to_proposal(:cp, current_cp)
# end

# function extract_collisions(tr)
#     t,_ = get_args(tr)
#     choices = get_choices(tr)
#     cols = Vector{Float64}(undef, t)
#     for i = 1:t
#         addr = :chain => i => :graph => :collision
#         cols[i] = choices[addr]
#     end
#     reshape(cols, (1,t))
# end

function cp_stats(state::Gen.ParticleFilterState)
    traces = Gen.get_traces(state)
    n = length(traces)
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
    println(p_cols[a:t])
    rejuv_cp = sum(p_cols[a:t]) >= 0.3
    if rejuv_cp & (t > 1)
        println("rejuv cp @ t $t")
        for i = 1:n
            trace = state.traces[i]
            cp = extract_cp(trace)
            if cp < t+1
                # (trace,_) = mh(trace, cp_proposal,
                #                (p_cols,), cp_involution,
                #                check = true)
                (trace,_) = mh(trace, congruency_proposal, (cp,),
                               congruency_involution)
                (trace,_) = mh(trace, incongruent_proposal, (cp,))
            end
            state.traces[i] = trace
        end
    end

    return nothing

end
