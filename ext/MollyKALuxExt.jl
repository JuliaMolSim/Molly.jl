# KernelAbstractions-based AEV computation for ANI ML potentials.
# Loaded when KernelAbstractions, Lux, and HDF5 are all in the environment.
# Provides GPU-portable @kernel implementations of the radial/angular AEV so the same
# code runs on CPU (via KA CPU backend) and CUDA/ROCm/Metal GPUs.
#
# BOUNDARY NOTE: these kernels compute neighbour displacements as raw coordinate
# differences (coords[j] - coords[i]); they do NOT apply the minimum-image convention.
# They are intended for non-periodic (gas-phase) systems, or where neighbour coordinates
# are already imaged. For periodic systems use the CPU path (potential_energy /
# compute_aevs), which applies vector(...) with the boundary.

module MollyKALuxExt

using Molly
import AtomsCalculators
using KernelAbstractions
using Lux, HDF5
using StaticArrays, LinearAlgebra

# The smooth cosine cutoff f_C ([ANI-1] Eq. 2) is `Molly.cosine_cutoff` (defined in core
# Molly, no Lux/HDF5 dependency) — a plain inlineable scalar function, device-compatible,
# reused by these kernels via `using Molly`.

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
        drj = coords[j] - ci
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for k in (j+1):n_atoms
            k == atom_i && continue
            drk = coords[k] - ci
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
        drj = coords[j] - ci
        rj  = norm(drj)
        rj >= r_c_A && continue
        fcj = cosine_cutoff(rj, r_c_A)
        sj  = species[j]

        for kk in (jj+1):hi
            k   = nbr_idx[kk]
            drk = coords[k] - ci
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
@kernel function _aev_kernel_wg!(
    aevs,
    @Const(coords),
    @Const(species),
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
            dr = coords[j] - ci
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
            drj = coords[j] - ci
            rj  = norm(drj)
            if rj < r_c_A
                fcj = cosine_cutoff(rj, r_c_A)
                sj  = species[j]
                for kk in (jj + 1):hi
                    k   = nbr_idx[kk]
                    drk = coords[k] - ci
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

# Convert a Molly NeighborList (flat half-pairs (i,j,special)) into a host CSR
# (offsets, symmetrised indices). This is the production path: it *consumes the
# neighbours produced by the system's neighbour finder* (CellListMapNeighborFinder on
# CPU, DistanceNeighborFinder on GPU) rather than building its own list. `neighbors.list`
# may be device-resident (GPU finder) — `Array(...)` brings the pairs to the host for the
# O(total_pairs) counting sort; offsets/indices are then uploaded to the compute backend.
function _nl_to_csr(neighbors, n_atoms::Int)
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

# Build a CSR neighbour list (offsets, indices) within `r_c_max` on the host.
# O(N²) scan — benchmarking fallback only (`neighbors=:auto`); production code should
# pass the system's NeighborList so Molly's neighbour finders drive the neighbour search.
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
        # Obtain a CSR neighbour list and move it to the compute backend.
        #   NeighborList  → consume the finder's neighbours (production path)
        #   :auto         → O(N²) host build from coords (benchmarking fallback)
        #   (off, idx)    → caller-supplied CSR
        off_h, idx_h = if neighbors === :auto
            _build_neighbor_csr(coords, max(r_c_R, r_c_A))
        elseif neighbors isa Tuple
            neighbors
        else
            _nl_to_csr(neighbors, n_atoms)
        end
        off_d = KernelAbstractions.allocate(ka_backend, Int32, length(off_h)); copyto!(off_d, off_h)
        idx_d = KernelAbstractions.allocate(ka_backend, Int32, length(idx_h)); copyto!(idx_d, idx_h)
        if write_reduce
            # One workgroup per atom; accumulate into a shared row, one global write per column.
            kernel! = _aev_kernel_wg!(ka_backend, workgroup)
            kernel!(
                aevs, coords, species, off_d, idx_d,
                η_R_d, r_s_R_d, r_c_R,
                η_A_d, r_s_A_d, θ_s_d, ζ, r_c_A,
                n_species, n_atoms,
                n_eta_R, n_shf_R, n_eta_A, n_shf_r, n_th, split, Val(aev_len);
                ndrange = n_atoms * workgroup,
            )
        else
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
    backend = nothing, neighbors = nothing,
)
    ka_backend = isnothing(backend) ? KernelAbstractions.get_backend(coords) : backend
    aevs = Molly.compute_aevs_ka(coords, species, pot.aev_params, n_species;
                                 backend = backend, neighbors = neighbors)   # (n_atoms, aev_len)
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

end # module MollyKALuxExt
