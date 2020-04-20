""" Defines a series of reversable jump proposal for changepoints"""


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
        for i = new_cp:old_cp
            addr = :chain => i => :graph => :collision
            @write_discrete_to_model(addr, true)
        end
        # @write_discrete_to_proposal(:cp, old_cp)
    else
        for i = old_cp:(new_cp - 1)
            addr = :chain => i => :graph => :collision
            @write_discrete_to_model(addr, false)
        end
        # @write_discrete_to_proposal(:cp, old_cp)
    end
    @write_discrete_to_proposal(:cp, old_cp)
end

function cp_stats(chain::Gen_Compose::SeqPFChain)

    parsed = current_step(chain)
    prop_collided = mean(parsed["unweighted"][:collision])





end
