using Gen
using Distributions
using Rotations

# Observation noise

struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const mat_noise = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end;

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    return mat
end;

# LogUniform proposals
struct LogUniform <: Gen.Distribution{Float64} end

const log_uniform = LogUniform()

function Gen.logpdf(::LogUniform, x::Float64, mu::U, noise::T) where {U<:Real,T<:Real}
    low = log(1 - noise)
    high = log(1 + noise)
    v = log(x) - log(mu)
    return (v >= low && v <= high) ? -log(high-low) : -Inf
end;

function Gen.random(::LogUniform, mu::U, noise::T) where {U<:Real,T<:Real}
    d = uniform(log(1 - noise), log(1 + noise))
    return exp(d + log(mu))
end;

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
