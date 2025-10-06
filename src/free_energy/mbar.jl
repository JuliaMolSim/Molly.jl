export 
    ThermoState,
    assemble_mbar_inputs,
    iterate_mbar,
    mbar_weights,
    pmf_with_uncertainty

const LOG_PREVFLOAT0 = log(nextfloat(0.0))      # ≈ log(nextfloat(0.0)) ≈ -744
const LOG_FLOATMAX   = log(floatmax(Float64))   # ≈ 709

# Thermodynamic state definition
struct ThermoState{B,P,S}
    name::String
    β::B         # 1 / (energy unit)
    p::P         # pressure (Quantity) or nothing
    system::S    # how to evaluate U_i on given coords+boundary
end

# Evaluate potential energy for a state i on a frame (coords, boundary)
@inline function _energy(sys::System{D, AT}, coords, boundary) where {D, AT}
    copyto!(sys.coords, coords)
    sys.boundary  = boundary
    return potential_energy(sys; n_threads = 1)
end

@inline function _as_energy_units(e, energy_units)
    try
        return uconvert(energy_units, e)                # e.g., kJ → kJ
    catch
        return uconvert(energy_units, e / 1u"mol")      # e.g., kJ → kJ/mol
    end
end

# Assemble K×N reduced potentials u for arbitrary states.
# coords_k      :: Vector{Vector{<:Any}}      per-window subsampled coordinates
# boundaries_k  :: Vector{Vector{<:Any}}      per-window subsampled boundaries (same length as coords_k[w])
# states        :: Vector{ThermoState}
# energy_units  :: Unitful energy unit used by Molly (e.g., u"kJ/mol")
#
# Returns: (u, N, win_of, shifts) and optionally u_target via helper below
function assemble_mbar_inputs(coords_k,
                              boundaries_k,
                              states::Vector{ThermoState};
                              target_state=nothing,
                              energy_units=u"kJ/mol")

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
        @inbounds for n in 1:N
            Uk   = _energy(sys, all_coords[n], all_boundaries[n])
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

function assemble_target_u(all_coords, all_boundaries, all_volumes, target::ThermoState;
                           energy_units=u"kJ/mol")
    N  = length(all_coords)
    βa = target.β
    pa = target.p
    has_p = pa !== nothing

    # single-state path; no threading to avoid per-thread system copies
    sys = target.system
    u_target = Vector{Float64}(undef, N)
    @inbounds for n in 1:N
        Ua   = _energy(sys, all_coords[n], all_boundaries[n])
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
    mbar_init(u::AbstractMatrix, win_of::AbstractVector{<:Integer};
              gauge_ref::Int=1, N_counts::Union{Nothing,AbstractVector{<:Integer}}=nothing) -> Vector{Float64}

Initializer per Eq. C4 (Shirts & Chodera 2008):
f_k^(0) = -mean( u[n, k] ) over frames n originating from state k.

Arguments
- u: K×N reduced-potential matrix (dimensionless), rows = states, cols = frames.
- win_of: length-N vector with origin window index (1..K) for each frame.
- gauge_ref: index of reference state to set f[gauge_ref] = 0.
- N_counts: optional length-K counts; checked against win_of if provided.

Returns
- f::Vector{Float64} of length K with gauge applied.
"""
function mbar_init(u::AbstractMatrix, win_of::AbstractVector{<:Integer};
                   gauge_ref::Int=1, N_counts::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    N, K = size(u)
    @assert length(win_of) == N "win_of length must equal number of columns in u"
    @assert 1 ≤ gauge_ref ≤ K "gauge_ref out of range"

    if N_counts !== nothing
        @assert length(N_counts) == K "N_counts must have length K"
        # verify counts match win_of
        counts = zeros(Int, K)
        @inbounds for n in 1:N
            k = win_of[n]
            @assert 1 ≤ k ≤ K "win_of contains invalid state index $k"
            counts[k] += 1
        end
        @assert counts == N_counts "win_of-derived counts differ from N_counts"
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
        @assert cnt > 0 "State $k has no frames"
        f[k] = -(sum_u / cnt)
    end

    ref = f[gauge_ref]
    @inbounds for k in 1:K
        f[k] -= ref
    end
    return f
end

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
    mbar_logD(u::AbstractMatrix, f::AbstractVector, logN::AbstractVector) -> Vector{Float64}

Compute log Dₙ = log ∑_k N_k exp(f_k − u[n,k]) for each frame n (log-sum-exp stable).
`u` is N×K: rows = frames n, cols = states k. `logN = log.(N_counts)`.
"""
function mbar_logD(u::AbstractMatrix, f::AbstractVector, logN::AbstractVector)
    N, K = size(u)
    @assert length(f) == K
    @assert length(logN) == K
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
    mbar_weights_sampled(u, f, logN; logD_n=nothing) -> (W, logD_n)

W[n,k] = exp( f[k] − u[n,k] − logD[n] ), with `u` as N×K. `logN` only enters Dₙ.
"""
function mbar_weights_sampled(u::AbstractMatrix, f::AbstractVector, logN::AbstractVector; logD_n=nothing)
    N, K = size(u)
    @assert length(f) == K == length(logN)
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
    mbar_weights_target(u_target, logD_n; shifts=nothing) -> w

wₙ ∝ exp( −(u_target[n] − sₙ) ) / Dₙ where `sₙ` are the same per-frame shifts applied to `u`.
Pass `shifts` you subtracted from `u`; if none were used, leave default.
"""
function mbar_weights_target(u_target::AbstractVector, logD_n::AbstractVector; shifts::Union{Nothing,AbstractVector}=nothing)
    @assert length(u_target) == length(logD_n)
    N = length(u_target)
    if shifts !== nothing
        @assert length(shifts) == N
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

function mbar_weights(u::AbstractMatrix,
                      u_target::AbstractVector,
                      f::AbstractVector,
                      logN::AbstractVector,
                      N_counts::AbstractVector;
                      shifts::Union{Nothing,AbstractVector}=nothing,
                      check::Bool=true)
    @assert all(isfinite.(u)) && all(isfinite.(u_target)) && all(isfinite.(f))
    @assert length(logN) == length(N_counts) && all(N_counts .> 0)
    logD = mbar_logD(u, f, logN)
    W, _ = mbar_weights_sampled(u, f, logN; logD_n=logD)
    w    = mbar_weights_target(u_target, logD; shifts=shifts)

    @assert all(isfinite.(W)) "Non-finite W_samp"
    @assert all(isfinite.(w)) "Non-finite w_target"
    any(iszero, W) && @warn "W_samp contains zeros; possible underflow"
    any(iszero, w) && @warn "w_target contains zeros; possible underflow"

    if check
        N, K = size(u)
        @inbounds for n in 1:N
            s = 0.0
            for k in 1:K
                s += N_counts[k] * W[n,k]
            end
            @assert isfinite(s) && abs(s - 1.0) ≤ 1e-8 "MBAR normalization failed at frame $n: sum_i N_i W[i,n] = $s"
        end
    end
    return W, w, logD
end

"""
    pmf_with_uncertainty(u, u_target, f, N_counts, logN, R_k;
                         nbins::Union{Int,Nothing}=nothing, edges=nothing, kBT=nothing,
                         zero::Symbol=:min, rmin=nothing, rmax=nothing)

Compute PMF along scalar CV r and its asymptotic (large-sample) uncertainty
using MBAR’s analytic covariance.

Returns NamedTuple:
(centers, F, F_energy, p, widths, edges, sigma_F, var_p)

If `edges` are not supplied, histogram bounds can be specified with
`rmin` and `rmax`. When both are omitted, the range defaults to
the min and max values of the collective variable. When `nbins` is not provided,
the bin count is estimated by the Freedman–Diaconis rule.

Assumes `u` is K×N reduced potentials with per-frame shifts already applied.
"""
function pmf_with_uncertainty(u::AbstractMatrix, u_target::AbstractVector,
                              f::AbstractVector, N_counts::AbstractVector,
                              logN::AbstractVector, R_k::Vector;
                              nbins::Union{Int,Nothing}=nothing, edges=nothing, kBT=nothing,
                              zero::Symbol=:min, rmin=nothing, rmax=nothing)

    N, K = size(u)
    @assert length(u_target) == N
    @assert length(f) == K
    @assert length(N_counts) == K
    @assert length(logN) == K
    @assert all(>(0), N_counts)

    # Flatten CV to match column order of u
    R_flat = Vector{eltype(first(R_k))}(undef, N)
    idx = 1
    for k in 1:K
        for r in R_k[k]
            R_flat[idx] = r
            idx += 1
        end
    end
    @assert idx == N + 1

    # Bin edges
    if edges === nothing
        if (rmin === nothing) != (rmax === nothing)
            error("rmin and rmax must both be provided or omitted")
        end
        if rmin === nothing
            rmin, rmax = extrema(R_flat)
        end
        @assert rmax > rmin "All r values identical; cannot bin"

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

        step = (rmax - rmin) / nbins
        edges = [rmin + i * step for i in 0:nbins]
    else
        edges = collect(edges)
    end
    nb = length(edges) - 1
    widths  = [edges[i+1] - edges[i] for i in 1:nb]
    centers = [(edges[i+1] + edges[i]) / 2 for i in 1:nb]
    Δr_inv  = 1.0 ./ Float64.(ustrip.(widths))  # for density

    # Sampled and target-state weights via mbar_weights (Eq. 9 and 13–15)
    W_sampled, W_na, logD = mbar_weights(u, u_target, f, logN, N_counts; check=false)
    W_samp = permutedims(W_sampled)  # N×K for concatenation with W_nA/W_na

    # Precompute vₙ = −u_target(n) − logDₙ for later use in ĉA (Eq. 14)
    v = @. -Float64(u_target) - logD

    @assert all(isfinite.(W_na)) "Non-finite values in W_na"
    any(iszero, W_na) && @warn "W_na contains zeros; possible underflow"

    # Prepare outputs
    p      = zeros(Float64, nb)      # bin probabilities under target
    var_p  = zeros(Float64, nb)      # asymptotic variance of p_i
    sigma_F = fill(NaN, nb)          # std dev of PMF in kBT

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
        # log ĉA via LSE on v restricted by mask
        if any(mask)
            mA = maximum(v[mask])
            sumexpA = sum(@. exp(v[mask] - mA))
            log_cA = mA + log(sumexpA)
            @. W_nA[mask] = exp(v[mask] - log_cA)
        else
            # Empty bin → keep zeros; variance undefined (stay NaN later)
            log_cA = -Inf
        end

        # Probability p_i = ĉA / ĉa = sum_n W_na * A_n (Eq. 15, 22)
        p[i] = dot(W_na, A)

        if p[i] == 0
            @warn "Zero probability in PMF bin $i: [$(edges[i]), $(edges[i+1]))"
        elseif p[i] > 0
            # Augmented W (N×(K+2)): sampled states, observable A, and target a
            @views W_aug[K + 1, :] .= W_nA
            @views W_aug[K + 2, :] .= W_na

            # Σ̂ for ln c’s via Eq. (D6): Σ̂ = ((WᵀW)^{-1} − N_aug)^+  (K+2 × K+2)
            G = W_aug * W_aug'
            Ginv = pinv(G)              # robust if not full rank
            H = Ginv - Matrix(N_aug)
            Σ = pinv(H)

            # Indices for A and a in augmented set
            idx_A = K + 1
            idx_a = K + 2

            # Var(p_i) from Eq. (16): p_i^2 ( Σ_AA + Σ_aa − 2 Σ_Aa )
            var_p[i] = p[i]^2 * (Σ[idx_A, idx_A] + Σ[idx_a, idx_a] - 2Σ[idx_A, idx_a])

            # PMF: F_i = −ln(p_i / Δr_i) up to constant (Eq. 23)
            # Uncertainty: Var(F_i) = Var(p_i) / p_i^2
            sigma_F[i] = sqrt(max(var_p[i], 0.0)) / p[i]
        end
    end

    if all(p .== 0)
        error("All bins have zero probability; chosen edges/rmin/rmax exclude all data")
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
    # Apply same shift to uncertainties? Not needed; sigma_F unaffected by additive constant.

    F_energy = isnothing(kBT) ? nothing : (F .* kBT)
    return (centers=centers, F=F, F_energy=F_energy, p=p, widths=widths, edges=edges,
            sigma_F=sigma_F, var_p=var_p)
end