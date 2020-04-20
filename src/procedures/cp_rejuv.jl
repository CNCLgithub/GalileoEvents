""" Defines a series of reversable jump proposal for changepoints"""


# TODO make t the cp
@gen function incongruent_proposal(tr::Gen.Trace)
    t, (_, prev_graph, _), _ = get_args(tr)
    choices = get_choices(tr)
    addr = :chain => cp => :physics => 1 => :persistence => :density
    if addr in choices
        prev_dens = choices[addr]
        {addr} ~ log_uniform(prev_dens, 0.2)
    end
end

""" Proposes changepoints

Only well defined for c < t and c == t
"""
@gen function cp_proposal(tr::Gen.Trace, ps::Vector{Float64})
    t, (_, prev_graph, _), _ = get_args(tr)
    state,graph,belief = last(ret_val(tr))
    col1 = first(prev_graph)
    col2 = first(graph)
    weights = zeros(t)
    if (!col1 & col2)
        # (cp == t) -> (cp < t)
        old_cp = t
        weights[:-1] = softmax(front(ps))
    else
        # (cp < t) -> (cp == t)
        old_cp = extract_cp(tr)
        weights[cp_old+1:] = softmax(ps[cp_old+1:])
    end
    new_cp = ({:cp} ~ categorical(weights))
    return (old_cp, new_cp)
end

@involution function cp_involution(model_args, proposal_args, proposal_retval)
    old_cp, new_cp = proposal_retval
    if new_cp < old_cp
        # (cp == t) -> (cp < t)
        addr = :chain => new_cp => :graph => :collision
        @write_discrete_to_model(addr, true)
        # for i = new_cp:old_cp
        #     @write_discrete_to_model(addr, true)
        # end
    else
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
    cols = Vector{Bool}(undef, t)
    for i = 1:t
        addr = :chain => i => :graph => :collision
        cols[i] = choices[addr]
    end
    reshape(cols, (1,t))
end

const latent_map = LatentMap(Dict(:collisions => extract_collisions))

function cp_stats(state::Gen.ParticleFilterState)
    traces = Gen.get_traces(state)
    n = length(traces)
    uw_traces = Gen.sample_unweighted_traces(state, n)
    unweighted = map(latent_map, uw_traces)
    estimates = merge(hcat, unweighted...)
    prop_collided = mean(estimates[:collision])
end


function cp_rejuv(proc::PopParticleFilter,
                  state::Gen.ParticleFilterState,
                  p_cols::Vector{Float64})
    n = length(state.traces)
    p_col = cp_stats(state)
    # TODO fix logic here
    min(p_col) < 0.5 && return p_col_t
    for i = 1:n
        (state.traces[i],_) = mh(state.traces[i], cp_proposal,
                                 (p_cols), cp_involution)
        (state.traces[i],_) = mh(state.traces[i], incongruent_proposal, tuple())
    end


end
