_tss_count(n::Integer, singular::AbstractString, plural::AbstractString=string(singular, "s")) =
    string(n, " ", n == 1 ? singular : plural)

mutable struct TSSStats{T}
    iterations::Vector{Int}
    active_state::Vector{Int}
    sampled_next_state::Vector{Int}
    max_abs_delta_f::Vector{T}
    f_history::Vector{Vector{T}}
    dens_history::Vector{Vector{T}}
    tilt_history::Vector{Vector{T}}
end

function Base.show(io::IO, stats::TSSStats)
    print(io, "TSSStats with ",
          _tss_count(length(stats.iterations), "logged entry", "logged entries"))
end

Base.show(io::IO, ::MIME"text/plain", stats::TSSStats) = show(io, stats)

function TSSStats(::Type{FT}) where {FT}
    return TSSStats{FT}(
        Int[],
        Int[],
        Int[],
        FT[],
        Vector{FT}[],
        Vector{FT}[],
        Vector{FT}[],
    )

end

function should_log_tss(iteration::Int, log_freq::Int)
    return iteration == 1 || (iteration % log_freq == 0)
end

function tss_vector_diagnostic(name, values, state)
    bad = findall(x -> !isfinite(x), values)
    n_show = min(length(bad), 8)
    shown_bad = bad[1:n_show]
    finite_values = filter(isfinite, collect(values))
    finite_msg = isempty(finite_values) ?
                 "no finite values" :
                 "finite range $(minimum(finite_values)) to $(maximum(finite_values))"

    return "TSS $(name) contains non-finite values at iteration $(state.iteration) " *
           "with active state $(state.active_state.active_idx); " *
           "bad indices $(shown_bad)$(length(bad) > n_show ? " ..." : ""); " *
           finite_msg
end

function check_tss_finite!(values, name::AbstractString, state)
    all(isfinite, values) || throw(ArgumentError(tss_vector_diagnostic(name, values, state)))
    return values
end

function check_tss_probabilities!(weights::AbstractVector{FT},
                                  name::AbstractString,
                                  state) where {FT}
    check_tss_finite!(weights, name, state)
    if any(<(zero(FT)), weights)
        throw(ArgumentError("TSS $(name) contains negative values at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end

    s = sum(weights)
    if !isfinite(s) || s <= zero(FT)
        throw(ArgumentError("TSS $(name) has invalid total weight $(s) at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    return weights
end

function check_tss_positive_probabilities!(weights::AbstractVector{FT},
                                           name::AbstractString,
                                           state) where {FT}
    check_tss_probabilities!(weights, name, state)
    if any(<=(zero(FT)), weights)
        throw(ArgumentError("TSS $(name) contains non-positive values at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    return weights
end

@inline function logaddexp_tss(a::T, b::T) where T
    if a == -T(Inf)
        return b
    elseif b == -T(Inf)
        return a
    end

    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

@inline function tss_log_update_arg(log_ratio::T, gain::T) where T
    gain == one(T) && return log_ratio
    return logaddexp_tss(log1p(-gain), log(gain) + log_ratio)
end

@inline tss_tilt_floor(::Type{FT}) where {FT} = sqrt(floatmin(FT))
