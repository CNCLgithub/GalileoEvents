using Gen
using Distributions
using Rotations

# Observation noise
struct RandomVec <: Gen.Distribution{Vector{Float64}} end

const random_vec = RandomVec()

function Gen.logpdf(::RandomVec, x::Vector{Float64}, mu::Vector{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end;

function Gen.random(::RandomVec, mu::Vector{U}, noise::T) where {U<:Real,T<:Real}
    vec = copy(mu)
    for i=1:length(mu)
        vec[i] = mu[i] + randn() * noise
    end
    return vec
end;
(::RandomVec)(mu, noise) = random(RandomVec(), mu, noise)

struct NoisyMatrix <: Gen.Distribution{Array{Float64}} end

const mat_noise = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Array{Float64}, mu::Array{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end;

function Gen.random(::NoisyMatrix, mu::Array{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    for i in CartesianIndeces(mu)
        mat[i] = mu[i] + randn() * noise
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


# struct Partial <: Gen.Distribution{T} where T end
# const partial{T} = Partial{T}()
# function Gen.random(::Partial{T}, mu::U) where {U<:Real}
#     d = Distributions.Truncated(Distributions.Normal(mu, noise),
#                                 low, high)
#     return Distributions.rand(d)
# end;
# function Gen.logpdf(::TruncNorm, x::Float64, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
#     d = Distributions.Truncated(Distributions.Normal(mu, noise),
#                                 low, high)
#     return Distributions.logpdf(d, x)
# end;
