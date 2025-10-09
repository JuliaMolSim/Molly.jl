export 
    ThermoState,
    assemble_mbar_inputs,
    iterate_mbar,
    mbar_weights,
    pmf_with_uncertainty

const LOG_PREVFLOAT0 = log(nextfloat(0.0))      # ≈ log(nextfloat(0.0)) ≈ -744
const LOG_FLOATMAX   = log(floatmax(Float64))   # ≈ 709

"""
    Represents a given thermodynamic state. It's just an abstraction over the
    [`System`](@ref) struct that carries the inverse temperatue (1/kBT) and the 
    pressure at which it was simulated.
"""
struct ThermoState{B,P,S}
    name::String
    β::B         # 1 / (energy unit)
    p::P         # pressure (Quantity) or nothing
    system::S    # how to evaluate U_i on given coords+boundary
end

function ThermoState(system::System, beta, press; name::Union{Nothing, String} = nothing)
    inv_ener = dimension(1/system.energy_units)
    if dimension(beta) != inv_ener
        throw(ArgumentError("β was not provided in appropriate dimensions: $(inv_ener)"))
    end
    if !isbar(press)
        throw(ArgumentError("Pressure was not provided in appropriate units"))
    end
    if name isa Nothing
        name = "system_$(beta)_$(pressure)"
    end
    return ThermoState(name, beta, press, system)
end

# Evaluate potential energy for a state i on a frame (coords, boundary)
@inline function _energy(sys::System, buffers, pe_vec_nounits, coords, boundary)
    copyto!(sys.coords, coords)
    sys.boundary  = boundary
    return potential_energy(sys, buffers, pe_vec_nounits, find_neighbors(sys); n_threads = 1)
end

@inline function _as_energy_units(e, energy_units)
    try
        return uconvert(energy_units, e)                # e.g., kJ → kJ
    catch
        return uconvert(energy_units, e / 1u"mol")      # e.g., kJ → kJ/mol
    end
end

"""
    assemble_mbar_inputs(coords_k, boundaries_k, states; target_state = nothing, energy_units = u"kJ/mol")

Assemble N×K reduced potentials u for arbitrary states.

Arguments:
- coords_k      :: Vector{Vector{<:Any}}      per-window subsampled coordinates
- boundaries_k  :: Vector{Vector{<:Any}}      per-window subsampled boundaries (same length as coords_k[w])
- states        :: Vector{ThermoState}        per-window thermodynamic states
- target_state  :: ThermoState                target thermodynamic states to compute resampling weights for
- energy_units  :: Unitful energy unit used by Molly (e.g., u"kJ/mol")

Returns: 
- (u=u, u_target=u_target, N = Nk, win_of=win_of, shifts=shifts)
"""
function assemble_mbar_inputs(coords_k,
                              boundaries_k,
                              states::Vector{ThermoState};
                              target_state::Union{Nothing, ThermoState{<:Any, <:Any, <:System{D, AT, T}}} = nothing,
                              energy_units=u"kJ/mol") where {D, AT, T}

    K = length(states)

    if length(coords_k) != K != length(boundaries_k)
        throw(ArgumentError("Length of coordinates / boundaries / states do not match"))
    end

    Nk = Int[length(coords_k[k]) for k in 1:K]
    N  = sum(Nk)

    # prefix sums
    starts = Vector{Int}(undef, K + 1)
    starts[1] = 1
    @inbounds for k in 1:K
        starts[k+1] = starts[k] + Nk[k]
    end
    ends = Vector{Int}(undef, K)
    @inbounds for k in 1:K
        ends[k] = starts[k+1] - 1
    end

    # window-of-frame
    win_of = Vector{Int}(undef, N)
    Threads.@threads for k in 1:K
        s = starts[k]; e = ends[k]
        @inbounds fill!(view(win_of, s:e), k)
    end

    # β and p
    β = [states[i].β for i in 1:K]
    p = Vector{Any}(undef, K)
    @inbounds for i in 1:K
        pi = states[i].p
        p[i] = (pi === nothing) ? nothing : pi
    end

    # flatten frames
    all_coords     = Vector{Any}(undef, N)
    all_boundaries = Vector{Any}(undef, N)
    all_volumes    = Vector{Any}(undef, N)
    Threads.@threads for idx in 1:N
        k = searchsortedlast(starts, idx)
        n = idx - starts[k] + 1
        ck = coords_k[k]
        bk = boundaries_k[k]
        b  = bk[n]
        @inbounds begin
            all_coords[idx]     = ck[n]
            all_boundaries[idx] = b
            all_volumes[idx]    = volume(b)
        end
    end

    # N×K reduced potentials: thread over states (columns)
    u = Matrix{Float64}(undef, N, K)
    Threads.@threads :static for k in 1:K
        sys = states[k].system              # private to this thread when k is unique
        βk  = β[k]
        pk  = p[k]
        # We initialize the buffers here to avoid copy overhead in GPU
        if AT <: AbstractGPUArray
            pe_vec_nounits = KernelAbstractions.zeros(get_backend(sys.coords), T, 1)
            buffers = init_buffers!(sys, 1, true)
        else
            pe_vec_nounits = nothing
            buffers = nothing
        end
        @inbounds for n in 1:N
            Uk   = _energy(sys, buffers, pe_vec_nounits, all_coords[n], all_boundaries[n])
            Uk_u = _as_energy_units(Uk, energy_units)
            red  = βk * Float64(ustrip(Uk_u))
            if pk !== nothing
                pV_u = _as_energy_units(pk * all_volumes[n], energy_units)
                red += βk * Float64(ustrip(pV_u))
            end
            u[n,k] = red                   # fill full column k
        end
    end

    # Row minima (per frame) and shift
    shifts = Vector{Float64}(undef, N)
    Threads.@threads for n in 1:N
        @inbounds shifts[n] = minimum(@view u[n, :])
    end
    @inbounds u .-= reshape(shifts, :, 1)

    u_target = (target_state === nothing) ? nothing :
               assemble_target_u(all_coords, all_boundaries, all_volumes, target_state; energy_units)

    return (u=u, u_target=u_target, N = Nk, win_of=win_of, shifts=shifts)
end

"""
    Assembles the reduced potentials vector for the target thermodynamic state
"""
function assemble_target_u(all_coords, all_boundaries, all_volumes, target::ThermoState{<:Any, <:Any, <:System{D,AT,T}};
                           energy_units=u"kJ/mol") where {D, AT, T}
    N  = length(all_coords)
    βa = target.β
    pa = target.p
    has_p = pa !== nothing

    # single-state path; no threading to avoid per-thread system copies
    sys = target.system
    u_target = Vector{Float64}(undef, N)

    if AT <: AbstractGPUArray
        pe_vec_nounits = KernelAbstractions.zeros(get_backend(sys.coords), T, 1)
        buffers = init_buffers!(sys, 1, true)
    else
        pe_vec_nounits = nothing
        buffers = nothing
    end

    @inbounds for n in 1:N
        Ua   = _energy(sys, buffers, pe_vec_nounits, all_coords[n], all_boundaries[n])
        Ua_u = _as_energy_units(Ua, energy_units)
        red  = βa * Float64(ustrip(Ua_u))
        if has_p
            pV_u = _as_energy_units(pa * all_volumes[n], energy_units)
            red += βa * Float64(ustrip(pV_u))
        end
        u_target[n] = red
    end
    return u_target
end


"""
Initializer per Eq. C4 (Shirts & Chodera 2008):
f_k^(0) = -mean( u[n, k] ) over frames n originating from state k.
"""
function mbar_init(u::AbstractMatrix, win_of::AbstractVector{<:Integer};
                   gauge_ref::Int=1, N_counts::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    N, K = size(u)
    if length(win_of) != N
        throw(ArgumentError("Length of win_of must equal number of columns in u"))
    end
    if !(1 ≤ gauge_ref ≤ K)
        throw(ArgumentError("gauge_ref out of range"))
    end

    if N_counts !== nothing
        if length(N_counts) != K
            throw(ArgumentError("N_counts must have length K"))
        end
        # verify counts match win_of
        counts = zeros(Int, K)
        @inbounds for n in 1:N
            k = win_of[n]
            if !(1 ≤ k ≤ K)
                throw(ArgumentError("win_of contains invalid state index $k"))
            end
            counts[k] += 1
        end
        if !(counts == N_counts)
            throw(ArgumentError("win_of-derived counts differ from N_counts"))
        end
    end

    f = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        sum_u = 0.0
        cnt   = 0
        for n in 1:N
            if win_of[n] == k
                sum_u += Float64(u[n, k])
                cnt   += 1
            end
        end
        if !(cnt > 0)
            throw(ArgumentError("State $k has no frames"))
        end
        f[k] = -(sum_u / cnt)
    end

    ref = f[gauge_ref]
    @inbounds for k in 1:K
        f[k] -= ref
    end
    return f
end

"""
Uses the self-consistent iteration method of Eq. C3 (Shirts & Chodera 2008)
to solve the MBAR equations.
"""
function mbar_iteration(u, f_init, logN)

    N, K = size(u)
    logD_n = mbar_logD(u, f_init, logN)

    f_new = Vector{Float64}(undef, K)
    Threads.@threads for k in 1:K
        # log-sum-exp over n of s_n = -u[n,k] - logD_n[n]
        m = -Inf
        @inbounds for n in 1:N
            v = -u[n,k] - logD_n[n]
            if v > m
                m = v
            end
        end
        sumexp = 0.0
        @inbounds for n in 1:N
            sumexp += exp(-u[n,k] - logD_n[n] - m)
        end
        f_new[k] = -(m + log(sumexp))
    end

    # gauge
    f_new .-= f_new[1]

    return f_new
end

"""
    iterate_mbar(u, win_of, N_counts; rtol = 1e-8, max_iter = 10_000)

    Solves the MBAR equations with the self-consistent iteration method described
    in Eq. C3 [Shirts & Chodera (2008)](https://doi.org/10.1063/1.2978177). 

Arguments:

- u        :: AbstractMatrix   N×K matrix or reduced potentials of the sampled N conformations under K thermodynamic states
- win_of   :: AbstractVector   vector of size N indexing the k ∈ K thermodynamic state the n ∈ N conformation was generated from
- N_counts :: AbstractVector   vector of size K enumerating the number of samples generated from each k ∈ K thermodynamic states

Returns:

- f_new :: AbstractVector   vector of size K with the relative free energy of each k ∈ K thermodynamic state  
- logN  :: AbstractVector   vector of size K with representing log(N_counts), useful to avoid downstream recomputations

"""
function iterate_mbar(u, win_of, N_counts; rtol::Float64=1e-8, max_iter::Int=10_000)
    logN  = log.(float.(N_counts))
    f_old = mbar_init(u, win_of; gauge_ref=1, N_counts=N_counts)
    f_new = mbar_iteration(u, f_old, logN)

    rel  = maximum(abs.(f_new .- f_old) ./ max.(abs.(f_old), 1.0))
    iter = 1

    pbar = ProgressBar(total = max_iter)

    while rel > rtol && iter < max_iter
        f_old = f_new
        f_new = mbar_iteration(u, f_old, logN)
        rel   = maximum(abs.(f_new .- f_old) ./ max.(abs.(f_old), 1.0))
        iter += 1
        update(pbar)
    end

    return f_new, logN
end

"""
Compute log Dₙ = log ∑_k N_k exp(f_k − u[n,k]) for each frame n (log-sum-exp stable).
`u` is N×K: rows = frames n, cols = states k. `logN = log.(N_counts)`.
"""
function mbar_logD(u::AbstractMatrix, f::AbstractVector, logN::AbstractVector)
    N, K = size(u)
    if length(f) != K
        throw(ArgumentError("Length of f must be equal to the number of thermodynamic states (K = $(K))"))
    end
    if length(logN) != K
        throw(ArgumentError("Length of logN must be equal to the number of thermodynamic states (K = $(K))"))
    end
    logD = Vector{Float64}(undef, N)
    @inbounds for n in 1:N
        m = -Inf
        @inbounds for k in 1:K
            s = logN[k] + f[k] - u[n,k]
            if s > m; m = s; end
        end
        se = 0.0
        @inbounds for k in 1:K
            se += exp(logN[k] + f[k] - u[n,k] - m)
        end
        logD[n] = m + log(se)
    end
    return logD
end

"""
W[n,k] = exp( f[k] − u[n,k] − logD[n] ), with `u` as N×K. `logN` only enters Dₙ.
"""
function mbar_weights_sampled(u::AbstractMatrix, f::AbstractVector, logN::AbstractVector; logD_n=nothing)
    N, K = size(u)
    if length(f) != K
        throw(ArgumentError("Length of f must be equal to the number of thermodynamic states (K = $(K))"))
    end
    if length(logN) != K
        throw(ArgumentError("Length of logN must be equal to the number of thermodynamic states (K = $(K))"))
    end
    logD = isnothing(logD_n) ? mbar_logD(u, f, logN) : logD_n
    W = Matrix{Float64}(undef, N, K)
    min_x =  Inf; max_x = -Inf
    min_i = 0; max_i = 0; min_n = 0; max_n = 0
    @inbounds for k in 1:K, n in 1:N
        x = f[k] - Float64(u[n,k]) - logD[n]
        if x < min_x; min_x = x; min_i = k; min_n = n; end
        if x > max_x; max_x = x; max_i = k; max_n = n; end
        W[n,k] = exp(x)
    end
    if min_x < LOG_PREVFLOAT0
        @warn "mbar_weights_sampled: underflow at (k=$min_i, n=$min_n): x=$min_x < $(LOG_PREVFLOAT0)"
    end
    if max_x > LOG_FLOATMAX
        throw(DomainError(max_x, "mbar_weights_sampled: overflow at (k=$max_i, n=$max_n)"))
    end
    return W, logD
end

"""
wₙ ∝ exp( −(u_target[n] − sₙ) ) / Dₙ where `sₙ` are the same per-frame shifts applied to `u`.
Pass `shifts` you subtracted from `u`; if none were used, leave default.
"""
function mbar_weights_target(u_target::AbstractVector, logD_n::AbstractVector; shifts::Union{Nothing,AbstractVector}=nothing)
    N = length(u_target)
    
    if N != length(logD_n)
        throw(ArgumentError("u_target must be of equal length as logD_n"))
    end
    
    if shifts !== nothing
        if length(shifts) != N
            throw(ArgumentError("shifts must be of equal length as u_target"))
        end
    end
    
    # vₙ = −(u_tgt[n] − sₙ) − logDₙ
    @inbounds begin
        m = -Inf
        for n in 1:N
            v = -Float64(u_target[n] - (shifts === nothing ? 0.0 : shifts[n])) - logD_n[n]
            if v > m; m = v; end
        end
        se = 0.0
        for n in 1:N
            se += exp(-Float64(u_target[n] - (shifts === nothing ? 0.0 : shifts[n])) - logD_n[n] - m)
        end
        logZ = m + log(se)

        w = Vector{Float64}(undef, N)
        min_x =  Inf; max_x = -Inf; min_n = 0; max_n = 0
        for n in 1:N
            x = -Float64(u_target[n] - (shifts === nothing ? 0.0 : shifts[n])) - logD_n[n] - logZ
            if x < min_x; min_x = x; min_n = n; end
            if x > max_x; max_x = x; max_n = n; end
            w[n] = exp(x)
        end
        if min_x < LOG_PREVFLOAT0
            @warn "mbar_weights_target: underflow at n=$min_n: x=$min_x < $(LOG_PREVFLOAT0)"
        end
        if max_x > LOG_FLOATMAX
            throw(DomainError(max_x, "mbar_weights_target: overflow at n=$max_n"))
        end
        return w
    end
end

"""
    mbar_weights(u, u_target, f, N_counts, logN; shifts = nothing, check = true)

    Calculates the weight matrix and target weights given a reduced energy matrix and
    target reduced potential vector, as per Eq. 13 in [Shirts & Chodera (2008)]](https://doi.org/10.1063/1.2978177).
    Allows the reweighting of arbitrary quantities to the target state.

Arguments:
- u        :: AbstractMatrix                             N×K matrix or reduced potentials of the sampled N conformations under K thermodynamic states
- u_target :: AbstractVector                             vector of length N with the reduced potential evaluated at each n ∈ N sampled conformation
- N_counts :: AbstractVector                             vector of lengthn k enumerating the number of samples generated by each k ∈ K thermodynamic state
- logN     :: AbstractVector                             vector of length K computed as the logarithm of the number of samples generated by each k ∈ K thermodynamic state
- shifts   :: Union{Nothing, AbstractVector} = nothing   if passed, vector of length N containing the numerical shifts used to prevent overflow in the computation of the free energies
- check    :: Bool = true                                flag detailing whether to perform sanity checks or not

Returns:
- w        :: AbstractMatrix   N×K matrix of the weights of each n ∈ N conformation for each k ∈ K thermodynamic state
- w_target :: AbstractVector   vector of length N with the weights for each n ∈ N conformation of the target thermodynamic state 
- logD     :: AbstractVector   vector of length N containing the longarithm of the denominator of Eq. 13, useful to avoid downstream recomputations

"""
function mbar_weights(u::AbstractMatrix,
                      u_target::AbstractVector,
                      f::AbstractVector,
                      N_counts::AbstractVector,
                      logN::AbstractVector;
                      shifts::Union{Nothing,AbstractVector}=nothing,
                      check::Bool=true)

    if !all(isfinite.(u))
        throw(DomainError(u, "Infintie values found in reduced potential"))
    end
    if !all(isfinite.(u_target))
        throw(DomainError(u_target, "Infintie values found in target system reduced potential"))
    end
    if !all(isfinite.(f))
        throw(DomainError(f, "Infintie values found in free energies"))
    end

    if length(logN) != length(N_counts)
        throw(DimensionMismatch("Length of logN and N_counts is not the same"))
    end
    if !all(N_counts .> 0)
        throw(DomainError(N_counts, "N_counts contains zeros"))
    end

    logD = mbar_logD(u, f, logN)
    W, _ = mbar_weights_sampled(u, f, logN; logD_n=logD)
    w    = mbar_weights_target(u_target, logD; shifts=shifts)

    if !all(isfinite.(W))
        throw(DomainError(W, "Non finite W_samp"))
    end
    if !all(isfinite.(w))
        throw(DomainError(w, "Non-finite w_target"))
    end

    any(iszero, W) && @warn "W_samp contains zeros; possible underflow"
    any(iszero, w) && @warn "w_target contains zeros; possible underflow"

    if check
        N, K = size(u)
        @inbounds for n in 1:N
            s = 0.0
            for k in 1:K
                s += N_counts[k] * W[n,k]
            end
            if !isfinite(s)
                throw(OverflowError("MBAR normalization failed at frame $n: sum_i N_i W[i,n] = $s"))
            end
        end
    end
    return W, w, logD
end

"""
    pmf_with_uncertainty(u, u_target, f, shifts, N_counts, logN, R_k;
                         nbins::Union{Int,Nothing}=nothing, edges=nothing, kBT=nothing,
                         zero::Symbol=:min, rmin=nothing, rmax=nothing)

Compute PMF along scalar CV r and its asymptotic (large-sample) uncertainty
using MBAR’s analytic covariance as per Eq. D8 of [Shirts &  Chodera (2008)](https://doi.org/10.1063/1.2978177).

Arguments:
- u        :: AbstractMatrix                             N×K matrix or reduced potentials of the sampled N conformations under K thermodynamic states
- u_target :: AbstractVector                             vector of length N with the reduced potential evaluated at each n ∈ N sampled conformation
- f        :: AbstractVector                             vector of length K containing the relative free energy of each k ∈ K thermodynamic state
- shifts   :: Union{Nothing, AbstractVector}             if passed, vector of length N containing the numerical shifts used to prevent overflow in the computation of the free energies
- N_counts :: AbstractVector                             vector of length K enumerating the number of samples generated by each k ∈ K thermodynamic state
- logN     :: AbstractVector                             vector of length K computed as the logarithm of the number of samples generated by each k ∈ K thermodynamic state
- R_k      :: Vector{AbstractVector}                     vector of length K in wich each element is a vector containing the sampled collective variable for each conformation generated by each k ∈ K thermodynamic state
- nbins    :: Union{Int, Nothing}            = nothing   the number of bins used to get the histogram of the collective variable
- edges    :: Union{AbstractVector, Nothing} = nothing   the edges of the histogram for the collective variable
- rmin     :: Union{Quantity, Nothing}       = nothing   minimum value for the histogram of the collective variale
- rmax     :: Union{Quantity, Nothing}       = nothing   maximum value for the histogram of the collective variale
- zero     :: Symbol                         = :min      how to gauge the potential of mean force, defaults to zero PMF to minimum value
- kBT      :: Union{Quantity, Nothing}       = nothing   kBT in appropriate units to get dimensional PMF

If `edges` are not supplied, histogram bounds can be specified with
`rmin` and `rmax`. When both are omitted, the range defaults to
the min and max values of the collective variable. When `nbins` is not provided,
the bin count is estimated by the Freedman–Diaconis rule.

Returns:
- NamedTuple: (centers, 
               F, 
               F_energy, 
               p, 
               widths, 
               edges, 
               sigma_F, 
               var_p)

"""
function pmf_with_uncertainty(u::AbstractMatrix, u_target::AbstractVector,
                              f::AbstractVector, shifts::AbstractVector,
                              N_counts::AbstractVector,
                              logN::AbstractVector, R_k::Vector;
                              nbins::Union{Int,Nothing} = nothing, edges = nothing, kBT = nothing,
                              zero::Symbol = :min, rmin = nothing, rmax = nothing)

    N, K = size(u)

    if length(u_target) != N
        throw(DimensionMismatch("length(u_target) = $(length(u_target)) must equal N = $N"))
    end
    if (length(f) != K ) || (length(N_counts) != K) || (length(logN) != K)
        throw(DimensionMismatch("lengths of f, N_counts, and logN must all equal K = $K"))
    end
    if !all(>(0), N_counts)
        throw(DomainError(N_counts, "All N_counts must be > 0"))
    end

    # Flatten CV to match column order of u
    R_flat = Vector{eltype(first(R_k))}(undef, N)
    idx = 1
    for k in 1:K
        for r in R_k[k]
            if idx > N
                throw(DimensionMismatch("R_k total length exceeds N = $N"))
            end
            R_flat[idx] = r
            idx += 1
        end
    end
    if idx != N + 1
        throw(DimensionMismatch("R_k total length $(idx-1) does not match N = $N"))
    end

    # Bin edges
    if edges === nothing
        if (rmin === nothing) != (rmax === nothing)
            throw(ArgumentError("rmin and rmax must both be provided or both omitted"))
        end
        if rmin === nothing
            rmin, rmax = extrema(R_flat)
        end
        if !(rmax > rmin)
            throw(DomainError((rmin, rmax), "All r values identical or invalid range; cannot bin"))
        end

        if nbins === nothing
            q75 = quantile(R_flat, 0.75)
            q25 = quantile(R_flat, 0.25)
            iqr = q75 - q25
            if iqr <= 0 || !isfinite(iqr)
                nbins = max(1, ceil(Int, sqrt(N)))
            else
                h = 2 * iqr / cbrt(N)
                nbins = max(1, ceil(Int, (rmax - rmin) / h))
            end
        end
        if nbins < 1
            throw(ArgumentError("nbins must be ≥ 1, got $nbins"))
        end

        step = (rmax - rmin) / nbins
        edges = [rmin + i * step for i in 0:nbins]
    else
        edges = collect(edges)
        if length(edges) < 2
            throw(ArgumentError("edges must have length ≥ 2"))
        end
        if !issorted(edges)
            throw(ArgumentError("edges must be sorted ascending"))
        end
        if any(!isfinite, edges)
            throw(DomainError(edges, "edges contain non-finite values"))
        end
    end

    nb = length(edges) - 1
    widths  = [edges[i+1] - edges[i] for i in 1:nb]
    if any(≤(0), widths)
        throw(DomainError(widths, "All bin widths must be > 0"))
    end
    centers = [(edges[i+1] + edges[i]) / 2 for i in 1:nb]
    Δr_inv  = 1.0 ./ Float64.(ustrip.(widths))  # for density

    # Sampled and target-state weights via mbar_weights (Eq. 9 and 13–15)
    W_sampled, W_na, logD = mbar_weights(u, u_target, f, N_counts, logN; shifts = shifts, check = false)
    rownorm = N_counts' * W_sampled'              # length-N vector of ∑k N_k W[n,k]
    W_sampled ./= rownorm'                        # enforce exact normalization

    W_samp = permutedims(W_sampled)  # K×N

    # Precompute vₙ = −u_target(n) − logDₙ for later use in ĉA (Eq. 14)
    s = isnothing(shifts) ? 0.0 : shifts
    v = @. -Float64(u_target) + s - logD

    if !all(isfinite.(W_na))
        throw(DomainError(W_na, "Non-finite values in W_na"))
    end
    any(iszero, W_na) && @warn "W_na contains zeros; possible underflow"

    # Prepare outputs
    p       = zeros(Float64, nb)      # bin probabilities under target
    var_p   = zeros(Float64, nb)      # asymptotic variance of p_i
    sigma_F = fill(NaN, nb)           # std dev of PMF in kBT

    # Diagonal N (augmented with zeros for {A,a})
    N_aug = Diagonal(vcat(Float64.(N_counts), 0.0, 0.0))

    # Precompute bin assignments for each frame
    bin_idx = searchsortedlast.(Ref(edges), R_flat)
    bin_idx[(bin_idx .== nb + 1) .& (R_flat .== edges[end])] .= nb

    # Scratch arrays for bin processing
    A     = Vector{Float64}(undef, N)
    W_nA  = similar(A)
    W_aug = Matrix{Float64}(undef, K + 2, N)
    @views W_aug[1:K, :] .= W_samp

    # Loop over bins, build augmented W = [W_samp  W_nA  W_na], then Σ̂ via Eq. (8)/(D6)
    for i in 1:nb
        fill!(A, 0)
        mask = bin_idx .== i
        A[mask] .= 1.0

        fill!(W_nA, 0)
        # ĉA = ∑_n A_n exp(−u_target(n)) / D_n  (Eq. 14)
        if any(mask)
            mA = maximum(v[mask])
            sumexpA = sum(@. exp(v[mask] - mA))
            log_cA = mA + log(sumexpA)
            @. W_nA[mask] = exp(v[mask] - log_cA)
        else
            log_cA = -Inf
        end

        # Probability p_i = sum_n W_na * A_n (Eq. 15, 22)
        p[i] = dot(W_na, A)

        if p[i] == 0
            @warn "Zero probability in PMF bin $i: [$(edges[i]), $(edges[i+1]))"
        elseif p[i] > 0
            @views W_aug[K + 1, :] .= W_nA
            @views W_aug[K + 2, :] .= W_na

            G   = W_aug * W_aug'
            Ginv = pinv(G)
            H   = Ginv - Matrix(N_aug)
            Σ   = pinv(H)

            idx_A = K + 1
            idx_a = K + 2

            var_p[i] = p[i]^2 * (Σ[idx_A, idx_A] + Σ[idx_a, idx_a] - 2Σ[idx_A, idx_a])

            sigma_F[i] = sqrt(max(var_p[i], 0.0)) / p[i]
        end
    end

    if all(p .== 0)
        throw(ArgumentError("All bins have zero probability; chosen edges/rmin/rmax exclude all data"))
    end

    # PMF values and zeroing
    pdens = p .* Δr_inv
    F = fill(NaN, nb)
    mask = pdens .> 0
    F[mask] .= -log.(pdens[mask])

    offset =
        zero === :last ? (let idx=findlast(mask); isnothing(idx) ? 0.0 : F[idx] end) :
        minimum(F[mask])

    F .-= offset

    bin_ess  = ess_per_bin(edges, R_flat, W_na)
    F_energy = isnothing(kBT) ? nothing : (F .* kBT)

    return (centers = centers, widths = widths, edges = edges,
            ess_per_bin = bin_ess,
            F = F, F_energy = F_energy, sigma_F = sigma_F,
            p = p, var_p = var_p)
end