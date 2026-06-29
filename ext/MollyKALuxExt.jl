# KernelAbstractions-based AEV computation for ANI ML potentials.
# Loaded when KernelAbstractions, Lux, and HDF5 are all in the environment.
# Provides GPU-portable @kernel implementations of _radial_aev! and _angular_aev!
# so the same code runs on CPU (via KA CPU backend) and CUDA/ROCm/Metal GPUs.

module MollyKALuxExt

using Molly
import AtomsCalculators
using KernelAbstractions
using Lux, HDF5
using StaticArrays, LinearAlgebra

# ============================================================================
# Scalar helpers (device-compatible — no closures, no allocation)
# ============================================================================

# Smooth cosine cutoff f_C — [ANI-1, Smith et al. Chem. Sci. 2017] Eq. (2).
@inline function _ka_cosine_cutoff(r::T, r_c::T) where T
    r < r_c ? T(0.5) * (one(T) + cos(T(π) * r / r_c)) : zero(T)
end

# ============================================================================
# KernelAbstractions AEV kernel — one thread per central atom
# ============================================================================

# Compute radial + angular AEV for atom `atom_i` from a dense distance table.
# GPU-portable form of _radial_aev! / _angular_aev! — same equations:
#   radial block  = [ANI-1] Eq. (3) G^R,   angular block = [ANI-1] Eq. (4) G^A.
# Uses the O(N²) all-pairs approach: each thread scans all n_atoms neighbours.
# For neighbour-list systems: call _aev_kernel_nl! instead (see below).
@kernel inbounds=true function _aev_kernel!(
    aevs,               # (n_atoms, aev_len) output matrix
    @Const(coords),     # (n_atoms,) SVector coordinates (Å)
    @Const(species),    # (n_atoms,) 1-based species index
    @Const(η_R),        # AEV radial params
    @Const(r_s_R),
    r_c_R,
    @Const(η_A),        # AEV angular params
    @Const(r_s_A),
    @Const(θ_s),
    ζ,
    r_c_A,
    n_species  :: Int,
    n_atoms    :: Int,
    n_eta_R    :: Int,
    n_shf_R    :: Int,
    n_eta_A    :: Int,
    n_shf_r    :: Int,
    n_th       :: Int,
    split      :: Int,    # boundary between radial/angular in AEV row
)
    atom_i = @index(Global, Linear)

    if atom_i <= n_atoms
    T       = eltype(coords[1])
    ci      = coords[atom_i]
    r_c_max = max(r_c_R, r_c_A)
    prefac0 = T(2)^(one(T) - ζ)

    # --- Radial AEV G^R, [ANI-1] Eq. (3) ---
    for j in 1:n_atoms
        j == atom_i && continue
        dr = coords[j] - ci
        r  = norm(dr)
        r >= r_c_R && continue
        fc   = _ka_cosine_cutoff(r, r_c_R)
        sj   = species[j]
        base = (sj - 1) * n_eta_R * n_shf_R
        for ki in 1:n_eta_R
            for kj in 1:n_shf_R
                aevs[atom_i, base + (ki-1)*n_shf_R + kj] +=
                    T(0.25) * exp(-η_R[ki] * (r - r_s_R[kj])^2) * fc
            end
        end
    end

    # --- Angular AEV G^A, [ANI-1] Eq. (4) ---
    for j in 1:n_atoms
        j == atom_i && continue
        drj = coords[j] - ci
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = _ka_cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for k in (j+1):n_atoms
            k == atom_i && continue
            drk = coords[k] - ci
            rk  = norm(drk)
            rk >= r_c_A && continue
            fck = _ka_cosine_cutoff(rk, r_c_A)
            sk  = species[k]

            s1, s2 = sj <= sk ? (sj, sk) : (sk, sj)
            pair_idx = (s1-1)*n_species - (s1-1)*(s1-2)÷2 + (s2-s1+1)

            r_avg  = (rj + rk) * T(0.5)
            fc_jk  = fcj * fck
            cos_th = clamp(dot(drj, drk) / (rj * rk), T(-1), T(1))
            theta  = acos(T(0.95) * cos_th)

            base = split + (pair_idx-1) * n_eta_A * n_shf_r * n_th
            for p in 1:n_eta_A
                for q in 1:n_shf_r
                    r_factor = prefac0 * fc_jk * exp(-η_A[p] * (r_avg - r_s_A[q])^2)
                    for l in 1:n_th
                        ang = (one(T) + cos(theta - θ_s[l]))^ζ
                        aevs[atom_i, base + ((p-1)*n_shf_r + (q-1))*n_th + l] += r_factor * ang
                    end
                end
            end
        end
    end
    end  # if atom_i <= n_atoms
end

# ============================================================================
# Neighbour-list kernel — each thread scans only atom_i's neighbours.
# Turns the all-pairs cost O(N²) radial / O(N³) angular into O(N·k) / O(N·k²),
# which is what makes large systems (≥1000 atoms) feasible on the GPU. The
# neighbour list is CSR-style: atom i's neighbours are nbr_idx[nbr_off[i]+1 :
# nbr_off[i+1]]. It is built within r_c_max = max(r_c_R, r_c_A); the per-term
# cutoff checks (r ≥ r_c_R / r_c_A) still select the right shell.
# ============================================================================
@kernel inbounds=true function _aev_kernel_nl!(
    aevs,
    @Const(coords),
    @Const(species),
    @Const(nbr_off),    # (n_atoms+1,) CSR offsets
    @Const(nbr_idx),    # (n_nbr_total,) flattened neighbour atom indices
    @Const(η_R),
    @Const(r_s_R),
    r_c_R,
    @Const(η_A),
    @Const(r_s_A),
    @Const(θ_s),
    ζ,
    r_c_A,
    n_species  :: Int,
    n_atoms    :: Int,
    n_eta_R    :: Int,
    n_shf_R    :: Int,
    n_eta_A    :: Int,
    n_shf_r    :: Int,
    n_th       :: Int,
    split      :: Int,
)
    atom_i = @index(Global, Linear)

    if atom_i <= n_atoms
    T       = eltype(coords[1])
    ci      = coords[atom_i]
    prefac0 = T(2)^(one(T) - ζ)
    lo      = nbr_off[atom_i] + 1
    hi      = nbr_off[atom_i + 1]

    # --- Radial AEV G^R, [ANI-1] Eq. (3) (over neighbours only) ---
    for jj in lo:hi
        j  = nbr_idx[jj]
        dr = coords[j] - ci
        r  = norm(dr)
        r >= r_c_R && continue
        fc   = _ka_cosine_cutoff(r, r_c_R)
        sj   = species[j]
        base = (sj - 1) * n_eta_R * n_shf_R
        for ki in 1:n_eta_R
            for kj in 1:n_shf_R
                aevs[atom_i, base + (ki-1)*n_shf_R + kj] +=
                    T(0.25) * exp(-η_R[ki] * (r - r_s_R[kj])^2) * fc
            end
        end
    end

    # --- Angular AEV G^A, [ANI-1] Eq. (4) (over neighbour pairs only) ---
    for jj in lo:hi
        j   = nbr_idx[jj]
        drj = coords[j] - ci
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = _ka_cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for kk in (jj+1):hi
            k   = nbr_idx[kk]
            drk = coords[k] - ci
            rk  = norm(drk)
            rk >= r_c_A && continue
            fck = _ka_cosine_cutoff(rk, r_c_A)
            sk  = species[k]

            s1, s2 = sj <= sk ? (sj, sk) : (sk, sj)
            pair_idx = (s1-1)*n_species - (s1-1)*(s1-2)÷2 + (s2-s1+1)

            r_avg  = (rj + rk) * T(0.5)
            fc_jk  = fcj * fck
            cos_th = clamp(dot(drj, drk) / (rj * rk), T(-1), T(1))
            theta  = acos(T(0.95) * cos_th)

            base = split + (pair_idx-1) * n_eta_A * n_shf_r * n_th
            for p in 1:n_eta_A
                for q in 1:n_shf_r
                    r_factor = prefac0 * fc_jk * exp(-η_A[p] * (r_avg - r_s_A[q])^2)
                    for l in 1:n_th
                        ang = (one(T) + cos(theta - θ_s[l]))^ζ
                        aevs[atom_i, base + ((p-1)*n_shf_r + (q-1))*n_th + l] += r_factor * ang
                    end
                end
            end
        end
    end
    end  # if atom_i <= n_atoms
end

# Build a CSR neighbour list (offsets, indices) within `r_c_max` on the host.
# O(N²) scan — fine for benchmarking; swap in a cell list for very large N.
function _build_neighbor_csr(coords::AbstractVector{SVector{D,T}}, r_c_max::T) where {D,T}
    n   = length(coords)
    rc2 = r_c_max^2
    off = Vector{Int32}(undef, n + 1)
    off[1] = 0
    @inbounds for i in 1:n
        ci = coords[i]; c = 0
        for j in 1:n
            j == i && continue
            sum(abs2, coords[j] - ci) < rc2 && (c += 1)
        end
        off[i+1] = off[i] + Int32(c)
    end
    idx = Vector{Int32}(undef, off[n+1])
    @inbounds for i in 1:n
        ci = coords[i]; pos = off[i]
        for j in 1:n
            j == i && continue
            if sum(abs2, coords[j] - ci) < rc2
                pos += 1
                idx[pos] = Int32(j)
            end
        end
    end
    return off, idx
end

# ============================================================================
# Public interface: compute_aevs_ka — GPU-portable AEV computation
# ============================================================================

"""
    Molly.compute_aevs_ka(coords, species_indices, p, n_species;
                          backend=nothing, neighbors=nothing, workgroup=256)

GPU-portable AEV computation using KernelAbstractions. One thread per atom.
`coords` can be a CPU Vector{SVector} or a GPU array (CuArray, ROCArray, etc.).
Pass `backend=MetalBackend()`/`CUDABackend()` to run on GPU; defaults to the
backend of `coords`.

`neighbors` selects the algorithm:
  * `nothing`  — O(N²)/O(N³) all-pairs kernel (only viable for tiny systems).
  * `:auto`    — build a CSR neighbour list on the host within `max(r_c_R, r_c_A)`
                 and run the O(N·k)/O(N·k²) neighbour-list kernel (use this ≥1000 atoms).
  * `(off, idx)` — a precomputed CSR neighbour list (Int32 offsets + indices).

`workgroup` sets the KA workgroup size (tunable per backend).

Returns the AEV matrix on the same device as `coords`.
"""
function Molly.compute_aevs_ka(
    coords  :: AbstractVector{SVector{D,T}},
    species :: AbstractVector{<:Integer},
    p,          # aev_params NamedTuple
    n_species :: Int;
    backend   = nothing,
    neighbors = nothing,
    workgroup = 256,
) where {D,T}
    n_atoms   = length(coords)
    n_eta_R   = length(p.η_R)
    n_shf_R   = length(p.r_s_R)
    n_eta_A   = length(p.η_A)
    n_shf_r   = length(p.r_s_A)
    n_th      = length(p.θ_s)
    n_pairs   = n_species * (n_species + 1) ÷ 2
    n_rad_per = n_eta_R * n_shf_R
    n_ang_per = n_eta_A * n_shf_r * n_th
    aev_len   = n_species * n_rad_per + n_pairs * n_ang_per
    split     = n_species * n_rad_per

    r_c_R = T(p.r_c_R)
    r_c_A = T(p.r_c_A)
    ζ     = T(p.ζ)

    # Allocate output on the same backend as coords
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(coords) : backend
    aevs = KernelAbstractions.zeros(ka_backend, T, n_atoms, aev_len)

    # Move parameter arrays to the same backend
    η_R_d   = KernelAbstractions.allocate(ka_backend, T, length(p.η_R)); copyto!(η_R_d,  p.η_R)
    r_s_R_d = KernelAbstractions.allocate(ka_backend, T, length(p.r_s_R)); copyto!(r_s_R_d, p.r_s_R)
    η_A_d   = KernelAbstractions.allocate(ka_backend, T, length(p.η_A)); copyto!(η_A_d,  p.η_A)
    r_s_A_d = KernelAbstractions.allocate(ka_backend, T, length(p.r_s_A)); copyto!(r_s_A_d, p.r_s_A)
    θ_s_d   = KernelAbstractions.allocate(ka_backend, T, length(p.θ_s)); copyto!(θ_s_d,  p.θ_s)

    if isnothing(neighbors)
        kernel! = _aev_kernel!(ka_backend, workgroup)
        kernel!(
            aevs, coords, species,
            η_R_d, r_s_R_d, r_c_R,
            η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
            n_species, n_atoms,
            n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split;
            ndrange = n_atoms,
        )
    else
        # Obtain a CSR neighbour list (build on host if requested) and move to device.
        off_h, idx_h = neighbors === :auto ?
            _build_neighbor_csr(coords, max(r_c_R, r_c_A)) : neighbors
        off_d = KernelAbstractions.allocate(ka_backend, Int32, length(off_h)); copyto!(off_d, off_h)
        idx_d = KernelAbstractions.allocate(ka_backend, Int32, length(idx_h)); copyto!(idx_d, idx_h)
        kernel! = _aev_kernel_nl!(ka_backend, workgroup)
        kernel!(
            aevs, coords, species, off_d, idx_d,
            η_R_d, r_s_R_d, r_c_R,
            η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
            n_species, n_atoms,
            n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split;
            ndrange = n_atoms,
        )
    end
    KernelAbstractions.synchronize(ka_backend)
    return aevs
end

end # module MollyKALuxExt
