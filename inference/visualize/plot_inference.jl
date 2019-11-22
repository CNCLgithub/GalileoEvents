using Gadfly
using Compose


function viz(results::Gen_Compose.StaticTraceResult)
    df = to_frame(results)
    # first the estimates
    estimates = map(x -> Gadfly.plot(df,
                                     x = x,
                                     color = :log_score,
                                     Gadfly.Geom.histogram),
                    results.latents)
    # last log scores
    log_scores = Gadfly.plot(df, x = :log_score, Gadfly.Geom.histogram)
    plot = vstack(estimates..., log_scores)
    compose(compose(context(), rectangle(), fill("white")), plot)
end

