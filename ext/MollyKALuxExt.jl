# KernelAbstractions-based AEV computation for ANI ML potentials.
# Loaded when KernelAbstractions, Lux, and HDF5 are all in the environment.
# Provides GPU-portable @kernel implementations of the radial/angular AEV so the same
# code runs on CPU (via KA CPU backend) and CUDA/ROCm/Metal GPUs.
#
# BOUNDARY NOTE: these kernels compute neighbour displacements with `Molly.vector(ci,
# coords[j], boundary)`, so they apply the minimum-image convention. Both `CubicBoundary`
# and `TriclinicBoundary` work (the boundary is only ever touched via `vector`). Pass the
# system boundary via the `boundary` kwarg of compute_aevs_ka/compute_ani_energy_ka; it is
# unit-stripped and converted to the coord element type. Omitting it (nothing) uses an
# infinite box, i.e. no minimum image (non-periodic / gas-phase).

module MollyKALuxExt

using Molly
import AtomsCalculators
using KernelAbstractions
using Lux, HDF5
using StaticArrays, LinearAlgebra

# The smooth cosine cutoff f_C ([ANI-1] Eq. 2) is `Molly.cosine_cutoff` (defined in core
# Molly, no Lux/HDF5 dependency) — a plain inlineable scalar function, device-compatible,
# reused by these kernels via `using Molly`.

# Produce an isbits boundary with element type T (matching the kernel's coords) for the
# minimum-image `vector(...)` calls. `nothing` → an infinite (non-periodic) box, which makes
# `vector` return the raw displacement. Metal is Float32-only, so the element type must match.
boundary_for_kernel(::Nothing, ::Val{D}, ::Type{T}) where {D, T} =
    CubicBoundary(SVector{D, T}(ntuple(_ -> T(Inf), D)))
boundary_for_kernel(b, ::Val{D}, ::Type{T}) where {D, T} =
    convert_boundary_eltype(Molly.strip_boundary(b), T)
convert_boundary_eltype(b::CubicBoundary, ::Type{T}) where T =
    CubicBoundary(SVector{3, T}(T.(b.side_lengths)))
convert_boundary_eltype(b::TriclinicBoundary{D, T2, C, A}, ::Type{T}) where {D, T2, C, A, T} =
    TriclinicBoundary(SVector(ntuple(i -> SVector{3, T}(T.(b.basis_vectors[i])), 3)); approx_images=A)

# ============================================================================
# KernelAbstractions AEV kernel — one thread per central atom
# ============================================================================

# Compute radial + angular AEV for atom `atom_i` from a dense distance table.
# GPU-portable form of radial_aev! / angular_aev! — same equations:
#   radial block  = [ANI-1] Eq. (3) G^R,   angular block = [ANI-1] Eq. (4) G^A.
# Uses the O(N²) all-pairs approach: each thread scans all n_atoms neighbours.
# For neighbour-list systems: call aev_kernel_nl! instead (see below).
@kernel inbounds=true function aev_kernel!(
    aevs,               # (n_atoms, aev_len) output matrix
    @Const(coords),     # (n_atoms,) SVector coordinates (Å)
    @Const(species),    # (n_atoms,) 1-based species index
    boundary,           # unit-stripped boundary (Cubic/Triclinic) for minimum image
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
        dr = Molly.vector(ci, coords[j], boundary)   # minimum-image displacement
        r  = norm(dr)
        r >= r_c_R && continue
        fc   = cosine_cutoff(r, r_c_R)
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
        drj = Molly.vector(ci, coords[j], boundary)
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for k in (j+1):n_atoms
            k == atom_i && continue
            drk = Molly.vector(ci, coords[k], boundary)
            rk  = norm(drk)
            rk >= r_c_A && continue
            fck = cosine_cutoff(rk, r_c_A)
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
@kernel inbounds=true function aev_kernel_nl!(
    aevs,
    @Const(coords),
    @Const(species),
    boundary,           # unit-stripped boundary (Cubic/Triclinic) for minimum image
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
        dr = Molly.vector(ci, coords[j], boundary)
        r  = norm(dr)
        r >= r_c_R && continue
        fc   = cosine_cutoff(r, r_c_R)
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
        drj = Molly.vector(ci, coords[j], boundary)
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for kk in (jj+1):hi
            k   = nbr_idx[kk]
            drk = Molly.vector(ci, coords[k], boundary)
            rk  = norm(drk)
            rk >= r_c_A && continue
            fck = cosine_cutoff(rk, r_c_A)
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
# Write-reduced kernel — ONE WORKGROUP per central atom.
# The W threads of the group split atom_i's neighbours (radial) and neighbour pairs
# (angular) and accumulate into a shared threadgroup row `acc` (aev_len floats, L passed
# as Val so it is a compile-time size) via atomic adds; the row is then written to global
# ONCE. This turns the ~O(nbrs²)·n_ang_per global read-modify-writes per atom of the
# one-thread-per-atom kernel into aev_len plain global writes, keeping the accumulation in
# fast threadgroup memory. Atomic order is nondeterministic → results match the scalar
# kernel only to Float32 rounding (~1e-6), not bit-for-bit.
@kernel function aev_kernel_wg!(
    aevs,
    @Const(coords),
    @Const(species),
    boundary,           # unit-stripped boundary (Cubic/Triclinic) for minimum image
    @Const(nbr_off),
    @Const(nbr_idx),
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
    ::Val{L},
) where {L}
    acc = @localmem Float32 (L,)

    # Phase 0 — zero the shared row cooperatively.
    Wz = @groupsize()[1]
    iz = @index(Local, Linear)
    while iz <= L
        acc[iz] = 0f0
        iz += Wz
    end
    @synchronize

    # Phase 1 — accumulate atom_i's AEV into the shared row. Indices are recomputed after
    # the barrier so this also runs on the KA CPU backend, which splits the kernel at
    # @synchronize and does not carry scalar locals across it (@localmem does persist).
    atom_i = @index(Group, Linear)
    lt     = @index(Local, Linear)
    W      = @groupsize()[1]
    if atom_i <= n_atoms
        T       = eltype(coords[1])
        ci      = coords[atom_i]
        prefac0 = T(2)^(one(T) - ζ)
        lo      = nbr_off[atom_i] + 1
        hi      = nbr_off[atom_i + 1]

        # Radial — threads stride over neighbours; [ANI-1] Eq. (3).
        jj = lo + (lt - 1)
        while jj <= hi
            j  = nbr_idx[jj]
            dr = Molly.vector(ci, coords[j], boundary)
            r  = norm(dr)
            if r < r_c_R
                fc   = cosine_cutoff(r, r_c_R)
                sj   = species[j]
                base = (sj - 1) * n_eta_R * n_shf_R
                for ki in 1:n_eta_R
                    for kj in 1:n_shf_R
                        KernelAbstractions.@atomic acc[base + (ki-1)*n_shf_R + kj] +=
                            T(0.25) * exp(-η_R[ki] * (r - r_s_R[kj])^2) * fc
                    end
                end
            end
            jj += W
        end

        # Angular — threads stride over the outer neighbour, inner runs to hi; [ANI-1] Eq. (4).
        jj = lo + (lt - 1)
        while jj <= hi
            j   = nbr_idx[jj]
            drj = Molly.vector(ci, coords[j], boundary)
            rj  = norm(drj)
            if rj < r_c_A
                fcj = cosine_cutoff(rj, r_c_A)
                sj  = species[j]
                for kk in (jj + 1):hi
                    k   = nbr_idx[kk]
                    drk = Molly.vector(ci, coords[k], boundary)
                    rk  = norm(drk)
                    if rk < r_c_A
                        fck = cosine_cutoff(rk, r_c_A)
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
                                    KernelAbstractions.@atomic acc[base + ((p-1)*n_shf_r + (q-1))*n_th + l] +=
                                        r_factor * ang
                                end
                            end
                        end
                    end
                end
            end
            jj += W
        end
    end
    @synchronize

    # Phase 2 — write the shared row to global once (coalesced across threads).
    atom_o = @index(Group, Linear)
    lto    = @index(Local, Linear)
    Wo     = @groupsize()[1]
    if atom_o <= n_atoms
        io = lto
        while io <= L
            aevs[atom_o, io] = acc[io]
            io += Wo
        end
    end
end

# ============================================================================
# Backward (adjoint) AEV kernels — one thread per central atom i. Given the AEV adjoint
# adj = ∂E/∂G, accumulate ∂E/∂r into `dcoords` (3, n_atoms) via atomic scatter to both
# endpoints of each neighbour pair. Displacements use vector(...) (minimum image), and the
# derivatives d/dr of f_C, the Gaussians and the angular term are analytic. Force = -∂E/∂r.
# ============================================================================

# Derivative of the cosine cutoff f_C wrt r: -0.5·(π/r_c)·sin(π·r/r_c) for r < r_c, else 0.
@inline cosine_cutoff_deriv(r::T, r_c::T) where T =
    r < r_c ? -T(0.5) * (T(π) / r_c) * sin(T(π) * r / r_c) : zero(T)

@kernel inbounds=true function aev_backward_radial!(
    dcoords,            # (3, n_atoms) accumulator for ∂E/∂r
    @Const(adj),        # (n_atoms, aev_len) AEV adjoint ∂E/∂G
    @Const(coords),
    @Const(species),
    boundary,
    @Const(nbr_off),
    @Const(nbr_idx),
    @Const(η_R),
    @Const(r_s_R),
    r_c_R,
    n_eta_R :: Int,
    n_shf_R :: Int,
    n_atoms :: Int,
)
    i = @index(Global, Linear)
    if i <= n_atoms
        T  = eltype(coords[1])
        ci = coords[i]
        lo = nbr_off[i] + 1
        hi = nbr_off[i + 1]
        for jj in lo:hi
            j  = nbr_idx[jj]
            dr = Molly.vector(ci, coords[j], boundary)
            r  = norm(dr)
            r >= r_c_R && continue
            u    = dr / r                       # unit vector along coords[j]-ci
            fc   = cosine_cutoff(r, r_c_R)
            fcp  = cosine_cutoff_deriv(r, r_c_R)
            sj   = species[j]
            base = (sj - 1) * n_eta_R * n_shf_R
            coeff = zero(T)                     # Σ adj·dG/dr for this pair's radial block
            for ki in 1:n_eta_R
                for kj in 1:n_shf_R
                    d     = r - r_s_R[kj]
                    e     = exp(-η_R[ki] * d * d)
                    dg_dr = T(0.25) * e * (fcp - T(2) * η_R[ki] * d * fc)
                    coeff += adj[i, base + (ki-1)*n_shf_R + kj] * dg_dr
                end
            end
            f1 = coeff * u[1]; f2 = coeff * u[2]; f3 = coeff * u[3]
            KernelAbstractions.@atomic dcoords[1, j] += f1
            KernelAbstractions.@atomic dcoords[2, j] += f2
            KernelAbstractions.@atomic dcoords[3, j] += f3
            KernelAbstractions.@atomic dcoords[1, i] += -f1
            KernelAbstractions.@atomic dcoords[2, i] += -f2
            KernelAbstractions.@atomic dcoords[3, i] += -f3
        end
    end
end

# Backward angular kernel. The angular term of atom i for neighbour pair (j,k) is
#   T = 2^(1-ζ)·f_C(r_ij)·f_C(r_ik)·exp(-η((r_avg-r_s)²))·(1+cos(θ-θ_s))^ζ,  [ANI-1] Eq. (4)
# a function of the displacements drj = r_j-r_i, drk = r_k-r_i only through r_avg=(rj+rk)/2,
# fc_jk = f_C(rj)f_C(rk) and θ = acos(0.95·cosθ). We accumulate ∂E/∂r into all three atoms by
# multivariate chain rule (∂θ/∂dr from ∂cosθ/∂dr; ∂r_avg/∂dr; ∂f_C/∂dr), atomic-scattering the
# equal-and-opposite reaction into atom i. Mirrors the forward aev_kernel_nl! angular loop.
@kernel inbounds=true function aev_backward_angular!(
    dcoords,            # (3, n_atoms) accumulator for ∂E/∂r (radial already added)
    @Const(adj),        # (n_atoms, aev_len) AEV adjoint ∂E/∂G
    @Const(coords),
    @Const(species),
    boundary,
    @Const(nbr_off),
    @Const(nbr_idx),
    @Const(η_A),
    @Const(r_s_A),
    @Const(θ_s),
    ζ,
    r_c_A,
    n_species :: Int,
    n_eta_A   :: Int,
    n_shf_r   :: Int,
    n_th      :: Int,
    split     :: Int,
    n_atoms   :: Int,
)
    i = @index(Global, Linear)
    if i <= n_atoms
        T       = eltype(coords[1])
        ci      = coords[i]
        prefac0 = T(2)^(one(T) - ζ)
        lo      = nbr_off[i] + 1
        hi      = nbr_off[i + 1]
        for jj in lo:hi
            j   = nbr_idx[jj]
            drj = Molly.vector(ci, coords[j], boundary)
            rj  = norm(drj)
            rj >= r_c_A && continue
            fcj   = cosine_cutoff(rj, r_c_A)
            fcj_p = cosine_cutoff_deriv(rj, r_c_A)
            sj    = species[j]
            uj    = drj / rj
            for kk in (jj + 1):hi
                k   = nbr_idx[kk]
                drk = Molly.vector(ci, coords[k], boundary)
                rk  = norm(drk)
                rk >= r_c_A && continue
                fck   = cosine_cutoff(rk, r_c_A)
                fck_p = cosine_cutoff_deriv(rk, r_c_A)
                sk    = species[k]
                uk    = drk / rk

                s1, s2   = sj <= sk ? (sj, sk) : (sk, sj)
                pair_idx = (s1-1)*n_species - (s1-1)*(s1-2)÷2 + (s2-s1+1)
                r_avg = (rj + rk) * T(0.5)
                fc_jk = fcj * fck
                c     = clamp(dot(drj, drk) / (rj * rk), T(-1), T(1))
                theta = acos(T(0.95) * c)
                base  = split + (pair_idx-1) * n_eta_A * n_shf_r * n_th

                # Reduce over (η_A, r_s_A, θ_s) into scalar sensitivities to r_avg / fc_jk / θ.
                SB = zero(T); SB_ravg = zero(T); SB_th = zero(T)
                for p in 1:n_eta_A
                    for q in 1:n_shf_r
                        d    = r_avg - r_s_A[q]
                        E_pq = exp(-η_A[p] * d * d)
                        coef = -T(2) * η_A[p] * d          # ∂E_pq/∂r_avg = E_pq·coef
                        B = zero(T); Bth = zero(T)
                        colbase = base + ((p-1)*n_shf_r + (q-1)) * n_th
                        for l in 1:n_th
                            A    = adj[i, colbase + l]
                            phi  = theta - θ_s[l]
                            ba   = one(T) + cos(phi)
                            ANG  = ba^ζ
                            dANG = ζ * ba^(ζ - one(T)) * (-sin(phi))
                            B   += A * ANG
                            Bth += A * dANG
                        end
                        SB      += E_pq * B
                        SB_ravg += E_pq * coef * B
                        SB_th   += E_pq * Bth
                    end
                end

                E_ravg = prefac0 * fc_jk * SB_ravg         # ∂E/∂r_avg
                E_fc   = prefac0 * SB                      # ∂E/∂fc_jk
                E_th   = prefac0 * fc_jk * SB_th           # ∂E/∂θ
                denom  = sqrt(max(one(T) - (T(0.95) * c)^2, T(1e-12)))
                tfac   = -T(0.95) / denom                  # ∂θ/∂(cosθ)

                a_rj = E_ravg * T(0.5) + E_fc * (fcj_p * fck)   # scalar coeff on uj
                a_rk = E_ravg * T(0.5) + E_fc * (fcj * fck_p)   # scalar coeff on uk
                tj   = E_th * tfac / rj                         # coeff on (uk - c·uj)
                tk   = E_th * tfac / rk                         # coeff on (uj - c·uk)

                dj1 = a_rj*uj[1] + tj*(uk[1] - c*uj[1])
                dj2 = a_rj*uj[2] + tj*(uk[2] - c*uj[2])
                dj3 = a_rj*uj[3] + tj*(uk[3] - c*uj[3])
                dk1 = a_rk*uk[1] + tk*(uj[1] - c*uk[1])
                dk2 = a_rk*uk[2] + tk*(uj[2] - c*uk[2])
                dk3 = a_rk*uk[3] + tk*(uj[3] - c*uk[3])

                KernelAbstractions.@atomic dcoords[1, j] += dj1
                KernelAbstractions.@atomic dcoords[2, j] += dj2
                KernelAbstractions.@atomic dcoords[3, j] += dj3
                KernelAbstractions.@atomic dcoords[1, k] += dk1
                KernelAbstractions.@atomic dcoords[2, k] += dk2
                KernelAbstractions.@atomic dcoords[3, k] += dk3
                KernelAbstractions.@atomic dcoords[1, i] += -(dj1 + dk1)
                KernelAbstractions.@atomic dcoords[2, i] += -(dj2 + dk2)
                KernelAbstractions.@atomic dcoords[3, i] += -(dj3 + dk3)
            end
        end
    end
end

# Convert a Molly NeighborList (flat half-pairs (i,j,special)) into a host CSR
# (offsets, symmetrised indices). This is the production path: it *consumes the
# neighbours produced by the system's neighbour finder* (CellListMapNeighborFinder on
# CPU, DistanceNeighborFinder on GPU) rather than building its own list. `neighbors.list`
# may be device-resident (GPU finder) — `Array(...)` brings the pairs to the host for the
# O(total_pairs) counting sort; offsets/indices are then uploaded to the compute backend.
function nl_to_csr(neighbors, n_atoms::Int)
    npairs = length(neighbors)                 # = neighbors.n
    host   = Array(neighbors.list)             # host copy (no-op if already on CPU)
    off = zeros(Int32, n_atoms + 1)
    cur = Vector{Int32}(undef, n_atoms)
    idx = Vector{Int32}(undef, 2 * npairs)
    @inbounds for ni in 1:npairs
        i, j, _ = host[ni]
        off[Int(i) + 1] += one(Int32)
        off[Int(j) + 1] += one(Int32)
    end
    @inbounds for a in 1:n_atoms
        off[a + 1] += off[a]
    end
    @inbounds for a in 1:n_atoms
        cur[a] = off[a]
    end
    @inbounds for ni in 1:npairs
        i, j, _ = host[ni]
        ii = Int(i); jj = Int(j)
        cur[ii] += one(Int32); idx[cur[ii]] = Int32(jj)
        cur[jj] += one(Int32); idx[cur[jj]] = Int32(ii)
    end
    return off, idx
end

# Build a CSR neighbour list (offsets, indices) within `r_c_max` on the host, using the
# minimum-image distance so it is correct under periodic boundaries. O(N²) scan —
# benchmarking fallback only (`neighbors=:auto`); production code should pass the system's
# NeighborList so Molly's neighbour finders drive the neighbour search.
function build_neighbor_csr(coords::AbstractVector{SVector{D,T}}, r_c_max::T, boundary) where {D,T}
    n   = length(coords)
    rc2 = r_c_max^2
    off = Vector{Int32}(undef, n + 1)
    off[1] = 0
    @inbounds for i in 1:n
        ci = coords[i]; c = 0
        for j in 1:n
            j == i && continue
            sum(abs2, Molly.vector(ci, coords[j], boundary)) < rc2 && (c += 1)
        end
        off[i+1] = off[i] + Int32(c)
    end
    idx = Vector{Int32}(undef, off[n+1])
    @inbounds for i in 1:n
        ci = coords[i]; pos = off[i]
        for j in 1:n
            j == i && continue
            if sum(abs2, Molly.vector(ci, coords[j], boundary)) < rc2
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
  * `nothing`     — O(N²)/O(N³) all-pairs kernel (only viable for tiny systems).
  * a `NeighborList` — consume the neighbours from the system's finder (production path;
                    `DistanceNeighborFinder` on GPU, `CellListMapNeighborFinder` on CPU) →
                    O(N·k)/O(N·k²) neighbour-list kernel. Preferred for ≥1000 atoms.
  * `:auto`       — O(N²) host build from coords, then the neighbour-list kernel
                    (benchmarking fallback only — does not use a real finder).
  * `(off, idx)`  — a precomputed CSR neighbour list (Int32 offsets + indices).

`workgroup` sets the KA workgroup size (tunable per backend).

`write_reduce=true` (with a neighbour list) uses the one-workgroup-per-atom kernel that
accumulates each atom's AEV row in shared threadgroup memory and writes it to global once,
instead of a global read-modify-write per term. Results match to Float32 rounding (~1e-6).

Returns the AEV matrix on the same device as `coords`.
"""
function Molly.compute_aevs_ka(
    coords  :: AbstractVector{SVector{D,T}},
    species :: AbstractVector{<:Integer},
    p,          # aev_params NamedTuple
    n_species :: Int;
    backend      = nothing,
    neighbors    = nothing,
    workgroup    = 256,
    write_reduce = false,
    boundary     = nothing,
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
    bdy   = boundary_for_kernel(boundary, Val(D), T)   # isbits, eltype T, for minimum image

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
        kernel! = aev_kernel!(ka_backend, workgroup)
        kernel!(
            aevs, coords, species, bdy,
            η_R_d, r_s_R_d, r_c_R,
            η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
            n_species, n_atoms,
            n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split;
            ndrange = n_atoms,
        )
    else
        # Obtain a CSR neighbour list and move it to the compute backend.
        #   NeighborList  → consume the finder's neighbours (production path)
        #   :auto         → O(N²) host build from coords (benchmarking fallback)
        #   (off, idx)    → caller-supplied CSR
        off_h, idx_h = if neighbors === :auto
            build_neighbor_csr(coords, max(r_c_R, r_c_A), bdy)
        elseif neighbors isa Tuple
            neighbors
        else
            nl_to_csr(neighbors, n_atoms)
        end
        off_d = KernelAbstractions.allocate(ka_backend, Int32, length(off_h)); copyto!(off_d, off_h)
        idx_d = KernelAbstractions.allocate(ka_backend, Int32, length(idx_h)); copyto!(idx_d, idx_h)
        if write_reduce
            # One workgroup per atom; accumulate into a shared row, one global write per column.
            kernel! = aev_kernel_wg!(ka_backend, workgroup)
            kernel!(
                aevs, coords, species, bdy, off_d, idx_d,
                η_R_d, r_s_R_d, r_c_R,
                η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
                n_species, n_atoms,
                n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split, Val(aev_len);
                ndrange = n_atoms * workgroup,
            )
        else
            kernel! = aev_kernel_nl!(ka_backend, workgroup)
            kernel!(
                aevs, coords, species, bdy, off_d, idx_d,
                η_R_d, r_s_R_d, r_c_R,
                η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
                n_species, n_atoms,
                n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split;
                ndrange = n_atoms,
            )
        end
    end
    KernelAbstractions.synchronize(ka_backend)
    return aevs
end

# ============================================================================
# End-to-end on-device ANI energy: GPU AEV + on-device element networks.
# The AEV matrix stays on the compute backend; the per-element Lux networks run on the
# same device (params moved once per (member, element)), so a full energy evaluation
# needs no host round-trip for the AEVs. Returns the ensemble-averaged energy in eV.
# ============================================================================
function Molly.compute_ani_energy_ka(
    coords, species, pot, n_species::Int;
    backend = nothing, neighbors = nothing, boundary = nothing,
)
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(coords) : backend
    aevs = Molly.compute_aevs_ka(coords, species, pot.aev_params, n_species;
                                 backend = backend, neighbors = neighbors,
                                 boundary = boundary)   # (n_atoms, aev_len)
    on_gpu = !(aevs isa Array)
    dev    = on_gpu ? Lux.gpu_device() : Lux.cpu_device()

    sp_host     = Array(species)
    idx_to_elem = Dict(v => k for (k, v) in pot.species_map)
    Ha_to_eV    = Molly.HARTREE_TO_EV

    # Per-species atom-index groups; uploaded to the compute device for the AEV gather.
    groups  = [Int32.(findall(==(s), sp_host)) for s in 1:n_species]
    idx_dev = map(groups) do g
        (on_gpu && !isempty(g)) || return g
        d = KernelAbstractions.allocate(ka_backend, Int32, length(g)); copyto!(d, g); d
    end

    n_ens = length(pot.ps_vec)
    E = 0.0
    for ens_i in 1:n_ens
        for s in 1:n_species
            g = groups[s]
            isempty(g) && continue
            sym   = Symbol(idx_to_elem[s])
            batch = permutedims(aevs[idx_dev[s], :])                  # (aev_len, n_s) on device
            ps_d  = dev(getfield(pot.ps_vec[ens_i], sym))
            st_d  = dev(getfield(pot.st_vec[ens_i], sym))
            out, _ = Lux.apply(getfield(pot.model, sym), batch, ps_d, st_d)  # (1, n_s) on device
            E += Float64(sum(out)) + Float64(pot.self_energies[s]) * length(g)
        end
    end
    return (E / n_ens) * Ha_to_eV
end

# ============================================================================
# On-device forces: reverse pass. F_k = -Σ_i (∂E_i/∂G_i)·(∂G_i/∂r_k). This part computes
# ∂E/∂G (the AEV adjoint) by a manual VJP through the element MLPs, all on-device (no
# Zygote/Enzyme on the GPU). The backward AEV kernels then turn ∂E/∂G into ∂E/∂r.
# ============================================================================

# Derivative of celu01 (α=0.1): 1 for z ≥ 0, else exp(z/0.1).
@inline celu01_deriv(z::T) where T = z >= zero(T) ? one(T) : exp(z / T(0.1))

# Manual forward + backward through one element's Lux.Chain of Dense layers (celu01 between,
# identity last). x is (in_dims, n_batch) on the compute device. Returns (Σ outputs, ∂E/∂x)
# where the loss is the sum of the scalar outputs over the batch.
function nn_energy_and_grad(model, ps, x)
    layers = values(model.layers)          # Dense layers, in order
    L = length(layers)
    zs = Vector{Any}(undef, L)             # pre-activations
    a  = x
    for k in 1:L
        W = ps[k].weight; b = ps[k].bias
        z = W * a .+ b
        zs[k] = z
        a = layers[k].activation === identity ? z : layers[k].activation.(z)
    end
    E = sum(a)                             # a is (1, n_batch)
    da = fill!(similar(a), one(eltype(a))) # ∂E/∂output = 1
    for k in L:-1:1
        W = ps[k].weight
        dz = layers[k].activation === identity ? da : da .* celu01_deriv.(zs[k])
        da = W' * dz                       # ∂E/∂a_{k-1}
    end
    return E, da                           # da is (in_dims, n_batch) = ∂E/∂x
end

# Ensemble-averaged energy (Hartree) and its gradient w.r.t. the AEVs, on-device.
# Returns (E_hartree, dE/dAEV) with dE/dAEV a (n_atoms, aev_len) array on the same backend.
function ani_energy_aev_grad_ka(aevs, species, pot, n_species::Int; backend=nothing)
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(aevs) : backend
    on_gpu = !(aevs isa Array)
    dev    = on_gpu ? Lux.gpu_device() : Lux.cpu_device()
    n_atoms, aev_len = size(aevs)
    sp_host     = Array(species)
    idx_to_elem = Dict(v => k for (k, v) in pot.species_map)

    groups  = [Int32.(findall(==(s), sp_host)) for s in 1:n_species]
    idx_dev = map(groups) do g
        (on_gpu && !isempty(g)) || return g
        d = KernelAbstractions.allocate(ka_backend, Int32, length(g)); copyto!(d, g); d
    end

    dEdAEV = KernelAbstractions.zeros(ka_backend, eltype(aevs), n_atoms, aev_len)
    n_ens  = length(pot.ps_vec)
    E = 0.0
    for ens_i in 1:n_ens
        for s in 1:n_species
            g = groups[s]; isempty(g) && continue
            sym   = Symbol(idx_to_elem[s])
            batch = permutedims(aevs[idx_dev[s], :])          # (aev_len, n_s)
            ps_d  = dev(getfield(pot.ps_vec[ens_i], sym))
            E_s, dEdbatch = nn_energy_and_grad(getfield(pot.model, sym), ps_d, batch)
            E += Float64(E_s) + Float64(pot.self_energies[s]) * length(g)
            @views dEdAEV[idx_dev[s], :] .+= permutedims(dEdbatch)   # scatter (unique rows)
        end
    end
    dEdAEV ./= n_ens
    return (E / n_ens), dEdAEV
end

# Given the AEV adjoint `adj` (n_atoms, aev_len) = ∂E/∂G, compute ∂E/∂r (3, n_atoms) on the
# backend by running the backward AEV kernels. Radial + angular. `neighbors`: NeighborList /
# :auto / (off,idx). `boundary` for the minimum image (matches the forward pass).
function aev_vjp_ka(adj, coords::AbstractVector{SVector{D,T}}, species, p, n_species::Int;
                    backend=nothing, neighbors=:auto, boundary=nothing, which=:both) where {D,T}
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(coords) : backend
    n_atoms = length(coords)
    n_eta_R = length(p.η_R); n_shf_R = length(p.r_s_R)
    n_eta_A = length(p.η_A); n_shf_r = length(p.r_s_A); n_th = length(p.θ_s)
    n_rad_per = n_eta_R * n_shf_R
    split     = n_species * n_rad_per
    r_c_R = T(p.r_c_R); r_c_A = T(p.r_c_A); ζ = T(p.ζ)
    bdy   = boundary_for_kernel(boundary, Val(D), T)

    off_h, idx_h = neighbors === :auto ? build_neighbor_csr(coords, max(r_c_R, r_c_A), bdy) :
                   neighbors isa Tuple ? neighbors : nl_to_csr(neighbors, n_atoms)
    off_d = KernelAbstractions.allocate(ka_backend, Int32, length(off_h)); copyto!(off_d, off_h)
    idx_d = KernelAbstractions.allocate(ka_backend, Int32, length(idx_h)); copyto!(idx_d, idx_h)

    η_R_d   = KernelAbstractions.allocate(ka_backend, T, n_eta_R); copyto!(η_R_d,  p.η_R)
    r_s_R_d = KernelAbstractions.allocate(ka_backend, T, n_shf_R); copyto!(r_s_R_d, p.r_s_R)
    η_A_d   = KernelAbstractions.allocate(ka_backend, T, n_eta_A); copyto!(η_A_d,  p.η_A)
    r_s_A_d = KernelAbstractions.allocate(ka_backend, T, n_shf_r); copyto!(r_s_A_d, p.r_s_A)
    θ_s_d   = KernelAbstractions.allocate(ka_backend, T, n_th);    copyto!(θ_s_d,  p.θ_s)

    dcoords = KernelAbstractions.zeros(ka_backend, T, 3, n_atoms)

    if which === :radial || which === :both
        rad! = aev_backward_radial!(ka_backend, 256)
        rad!(dcoords, adj, coords, species, bdy, off_d, idx_d,
             η_R_d, r_s_R_d, r_c_R, n_eta_R, n_shf_R, n_atoms; ndrange = n_atoms)
        KernelAbstractions.synchronize(ka_backend)
    end

    if which === :angular || which === :both
        ang! = aev_backward_angular!(ka_backend, 256)
        ang!(dcoords, adj, coords, species, bdy, off_d, idx_d,
             η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A, n_species,
             n_eta_A, n_shf_r, n_th, split, n_atoms; ndrange = n_atoms)
        KernelAbstractions.synchronize(ka_backend)
    end

    return dcoords
end

# End-to-end on-device ANI forces (eV/Å). Forward AEV → NN VJP (∂E/∂G) → backward AEV kernels
# (∂E/∂r) → F = -∂E/∂r · Ha→eV, ensemble-averaged. This is the single ANI forces path — the
# AtomsCalculators.forces! method below dispatches here for both CPU (KA CPU backend) and GPU
# (Metal/CUDA) systems. The former finite-difference and Enzyme reverse-mode paths are retired:
# the analytic backward is both exact (matches TorchANI/FD to ~1e-6) and, on CPU, 7–14× faster.
function Molly.compute_ani_forces_ka(coords, species, pot, n_species::Int;
        backend=nothing, neighbors=nothing, boundary=nothing)
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(coords) : backend
    aevs = Molly.compute_aevs_ka(coords, species, pot.aev_params, n_species;
        backend=backend, neighbors=neighbors, boundary=boundary)
    _, dEdAEV = ani_energy_aev_grad_ka(aevs, species, pot, n_species; backend=ka_backend)
    bwd_nb  = isnothing(neighbors) ? :auto : neighbors
    dcoords = aev_vjp_ka(dEdAEV, coords, species, pot.aev_params, n_species;
        backend=ka_backend, neighbors=bwd_nb, boundary=boundary)   # ∂E_Ha/∂r, (3, n_atoms)
    T    = eltype(eltype(coords))
    Ha   = T(Molly.HARTREE_TO_EV)
    fmat = Array(dcoords)
    return [SVector{3,T}(-fmat[1,i]*Ha, -fmat[2,i]*Ha, -fmat[3,i]*Ha) for i in 1:length(coords)]
end

# The single ANI forces path. Strips coords to Å (staying on their device — Array → KA CPU,
# GPU array → Metal/CUDA), computes forces via the analytic backward, and accumulates into `fs`
# in the system's units. Replaces the former finite-difference (MollyLuxExt) and Enzyme
# reverse-mode (MollyLuxEnzymeExt) methods. Requires KernelAbstractions in addition to Lux/HDF5.
function AtomsCalculators.forces!(fs, sys::System{D, AT, T}, inter::ANIPotential;
                                  kwargs...) where {D, AT, T}
    nbrs    = get(kwargs, :neighbors, nothing)
    n_sp    = length(inter.species_map)
    sp      = Int32[inter.species_map[ad.element] for ad in sys.atoms_data]
    coords  = Molly.ustrip_vec.(u"Å", sys.coords)       # Å, unitless; stays on coords' device
    species = Molly.to_device(sp, AT)
    backend = KernelAbstractions.get_backend(coords)
    F = Molly.compute_ani_forces_ka(coords, species, inter, n_sp;                 # host, eV/Å
            backend = backend, neighbors = nbrs, boundary = Molly.strip_boundary(sys.boundary))
    if AT <: Array
        @inbounds for i in eachindex(fs)
            fs[i] += Molly.ani_force_to_units(SVector{D, T}(F[i]), sys.force_units, AT)
        end
    else   # GPU system: build the unit-carrying increment on the host, then add on-device
        inc = [Molly.ani_force_to_units(SVector{D, T}(F[i]), sys.force_units, Array)
               for i in eachindex(F)]
        fs .+= Molly.to_device(inc, AT)
    end
    return fs
end

end # module MollyKALuxExt
