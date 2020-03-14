using Gadfly
using Compose
import Cairo


function plot_extract(e::Dict)
    latents = collect(keys(e["weighted"]))
    df = _to_frame(e["log_scores"], e["unweighted"])
    # first the estimates
    estimates = map(y -> Gadfly.plot(df,
                                     y = y,
                                     x = :t,
                                     color = :log_score,
                                     Gadfly.Geom.histogram2d),
                    latents)
    # last log scores
    log_scores = Gadfly.plot(df, y = :log_score, x = :t,
                             Gadfly.Geom.histogram2d)
    plot = vstack(estimates..., log_scores)
    compose(compose(context(), rectangle(), fill("white")), plot)
end
