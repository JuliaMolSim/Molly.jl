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

@inline function _ka_cosine_cutoff(r::T, r_c::T) where T
    r < r_c ? T(0.5) * (one(T) + cos(T(π) * r / r_c)) : zero(T)
end

# ============================================================================
# KernelAbstractions AEV kernel — one thread per central atom
# ============================================================================

# Compute radial + angular AEV for atom `atom_i` from a dense distance table.
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

    # --- Radial AEV ---
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

    # --- Angular AEV ---
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
# Public interface: compute_aevs_ka — GPU-portable AEV computation
# ============================================================================

"""
    Molly.compute_aevs_ka(coords, species_indices, p, n_species; backend=nothing)

GPU-portable AEV computation using KernelAbstractions. One thread per atom.
`coords` can be a CPU Vector{SVector} or a GPU array (CuArray, ROCArray, etc.).
Pass `backend=CUDABackend()` to run on NVIDIA GPU; defaults to CPU() backend.

Returns the AEV matrix on the same device as `coords`.
"""
function Molly.compute_aevs_ka(
    coords  :: AbstractVector{SVector{D,T}},
    species :: AbstractVector{<:Integer},
    p,          # aev_params NamedTuple
    n_species :: Int;
    backend = nothing,
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

    kernel! = _aev_kernel!(ka_backend, 256)
    kernel!(
        aevs, coords, species,
        η_R_d, r_s_R_d, r_c_R,
        η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
        n_species, n_atoms,
        n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split;
        ndrange = n_atoms,
    )
    KernelAbstractions.synchronize(ka_backend)
    return aevs
end

end # module MollyKALuxExt
