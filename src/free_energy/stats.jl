export 
    statistical_inefficiency,
    subsample

struct StatisticalInefficiency{I, S, L, E, LG}
    inefficiency::I
    stride::S
    input_length::L
    effective_size::E
    lag::LG
end

@doc """
    statistical_inefficiency(series::AbstractVector; maxlag::Union{Nothing,Int}=nothing)

Integrated autocorrelation time estimator with IPS truncation and finite-sample taper.
Returns a NamedTuple: (g, stride, N, N_eff, L).
- g: statistical inefficiency
- stride: ceil(Int, g)
- N: input length
- N_eff: floor(N / stride)
- L: truncation lag used

Notes:
- Uses initial positive sequence (IPS) on paired lags to choose L.
- Uses normalized ACF of the mean-removed series.
- Includes the (1 - τ/N) taper in the sum.
"""
function statistical_inefficiency(series::AbstractVector; maxlag::Union{Nothing,Int}=nothing)
    x = ustrip.(series)              # remove units if present; else no-op if plain floats
    N = length(x)
    if N < 3
        return (g = 1.0, stride = 1, N = N, N_eff = N, L = 0)
    end

    μ = mean(x)
    @inbounds @simd for i in eachindex(x)
        x[i] -= μ
    end

    s2 = sum(abs2, x) / (N - 1)                # sample variance
    if !isfinite(s2) || s2 == 0.0
        return (g = 1.0, stride = 1, N = N, N_eff = N, L = 0)
    end

    Lmax = isnothing(maxlag) ? min(N - 1, fld(N, 2)) : min(maxlag, N - 1)
    C = Vector{Float64}(undef, Lmax)

    # normalized autocorrelation for lags 1..Lmax
    @inbounds @views for lag in 1:Lmax
        num = dot(x[1:N-lag], x[1+lag:N])
        C[lag] = num / ((N - lag) * s2)
    end

    # Initial Positive Sequence (pairwise) to choose truncation L
    L = 0
    M = fld(Lmax, 2)
    @inbounds for k in 1:M
        gamma = C[2k - 1] + C[2k]
        if gamma > 0
            L = 2k
        else
            break
        end
    end
    if L == 0
        idx = findfirst(c -> c <= 0.0, C)
        L = isnothing(idx) ? Lmax : max(idx - 1, 1)
    end

    # Integrated ACF with finite-sample taper
    wsum = 0.0
    @inbounds for τ in 1:L
        wsum += (1 - τ / N) * C[τ]
    end
    g = max(1.0, 1 + 2 * wsum)
    stride = max(1, ceil(Int, g))
    N_eff = max(1, fld(N, stride))
    return StatisticalInefficiency(g, stride, N, N_eff, L)
end

function subsample(series::AbstractVector, stride::Int; first::Int = 1)
    return series[first:stride:end]
end

# ESS of a mask (no views with Bool indexing)
@inline function ess_mask(mask::AbstractVector{Bool}, w::AbstractVector)
    s, ssq = 0.0, 0.0
    @inbounds @simd for i in eachindex(mask, w)
        if mask[i]
            wi = float(w[i]); s += wi; ssq += wi*wi
        end
    end
    (s>0 && ssq>0) ? (s*s/ssq) : 0.0
end

# Per-partition ESS
function ess_per_bin(edges::AbstractVector, r::AbstractVector, w::AbstractVector)
    nb = length(edges)-1
    idx = searchsortedlast.(Ref(edges), r)
    idx[(idx .== nb+1) .& (r .== edges[end])] .= nb
    e = Vector{Float64}(undef, nb)
    @inbounds for i in 1:nb
        e[i] = ess_mask(idx .== i, w)
    end
    e
end