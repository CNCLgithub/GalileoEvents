using Distributions

export mat_noise,
    log_uniform,
    trunc_norm

struct NoisyMatrix <: Gen.Distribution{Array{Float64}} end

const mat_noise = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Array{Float64}, mu::Array{<:Real}, noise::T) where {T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end;

function Gen.random(::NoisyMatrix, mu::Array{<:Real}, noise::T) where {T<:Real}
    mat = copy(mu)
    for i in CartesianIndices(mu)
        mat[i] = mu[i] + randn() * noise
    end
    return mat
end;

# LogUniform proposals


struct LogUniform <: Gen.Distribution{Float64} end

const log_uniform = LogUniform()

function Gen.logpdf(::LogUniform, x::Float64, low::T, high::T) where {T<:Real}
    l = log(low)
    h = log(high)
    v = log(x)
    # println("$v ($x) | $l ($low) , $h ($high)")
    return (v >= l && v <= h) ? -log(h-l) : -Inf
end

function Gen.random(::LogUniform, low::T, high::T) where {T<:Real}
    d = uniform(log(low), log(high))
    exp(d)
end

# Truncated Distributions
struct TruncNorm <: Gen.Distribution{Float64} end
const trunc_norm = TruncNorm()
function Gen.random(::TruncNorm, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
    d = Distributions.Truncated(Distributions.Normal(mu, noise),
                                low, high)
    return Distributions.rand(d)
end;
function Gen.logpdf(::TruncNorm, x::Float64, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
    d = Distributions.Truncated(Distributions.Normal(mu, noise),
                                low, high)
    return Distributions.logpdf(d, x)
end;

