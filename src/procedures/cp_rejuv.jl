""" Defines a series of reversable jump proposal for changepoints"""


@gen function cp_proposal(tr::Gen.Trace, ps::Vector{Float64})
    t, (_, prev_graph, _), _ = get_args(tr)
    state,graph,belief = last(ret_val(tr))
    col1 = first(prev_graph)
    col2 = first(graph)
    if (!col1 & col2)
        # (cp == t) -> (cp < t)
        old_cp = t
        weights = softmax(front(ps))
    # else if
    #     # changepoint after time t
    else
        # (cp < t) -> (cp == t)
        old_cp = extract_cp(tr)
        weights = softmax(ps[cp_old+1:end])
    end
    new_cp = ({:cp} ~ categorical(weights))
    return (old_cp, new_cp)
end

@involution function cp_involution(model_args, proposal_args, proposal_retval)
    old_cp, new_cp = proposal_retval

end

function cp_stats(chain::Gen_Compose::SeqPFChain)

    parsed = current_step(chain)
    prop_collided = mean(parsed["unweighted"][:collision])





end
