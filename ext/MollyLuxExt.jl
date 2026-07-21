# ANI (and future ML potential) support via Lux.jl + HDF5.jl
# Loaded when both Lux and HDF5 are in the user environment.
#
# ===========================================================================
# How ANI works (and where each equation lives in this file)
# ===========================================================================
# References:
#   [ANI-1]  J. S. Smith, O. Isayev, A. E. Roitberg, "ANI-1: an extensible neural
#            network potential with DFT accuracy at force field computational cost",
#            Chem. Sci. 2017, 8, 3192–3203.  doi:10.1039/C6SC05720A
#   [ANI-2x] C. Devereux et al., "Extending the Applicability of the ANI Deep Learning
#            Molecular Potential to Sulfur and Halogens", J. Chem. Theory Comput. 2020,
#            16, 4192–4202.  doi:10.1021/acs.jctc.0c00121   (adds S, F, Cl; same AEV form)
#   [ANI-1x] J. S. Smith et al., "Less is more: Sampling chemical space with active
#            learning", J. Chem. Phys. 2018, 148, 241733.  doi:10.1063/1.5023802
#            (the ensemble model — energy is averaged over M networks; ANI-2x uses M=8)
#
# The total potential energy is a sum of per-atom contributions, [ANI-1] Eq. (1):
#       E = Σ_i E_i
# Each atom i is described by an Atomic Environment Vector (AEV) — a fixed-length
# fingerprint of its local chemical environment built from modified Behler–Parrinello
# symmetry functions. The AEV is fed to a small element-specific neural network that
# outputs that atom's energy E_i; ANI-2x averages 8 such networks (an ensemble).
#
#   cosine_cutoff   → [ANI-1] Eq. (2)  smooth radial cutoff f_C(R_ij)
#   radial_aev!    → [ANI-1] Eq. (3)  radial symmetry function  G^R
#   angular_aev!   → [ANI-1] Eq. (4)  angular symmetry function G^A
#   ani_energy_single / ani_raw_energy → [ANI-1] Eq. (1) energy sum + ensemble average
# The element NN architecture is in build_element_model; per-atom self-energies are an
# additive reference shift (Hartree). This extension provides the AEV + energy and the analytic
# on-device forces (AtomsCalculators.forces! → compute_ani_forces_ka).

module MollyLuxExt

using Molly
using Molly: from_device, to_device, vector
import AtomsCalculators
using Lux, HDF5
using KernelAbstractions   # strong Molly dependency; the ANI GPU kernels live in molly_lux_ka.jl
using StaticArrays, Unitful, Random, LinearAlgebra

# ============================================================================
# Shared infrastructure
# ============================================================================

# Strip units from a coordinate SVector array and return plain Float SVectors.
function strip_coords(coords::AbstractVector{SVector{D,T}}) where {D,T}
    unit(first(coords)[1]) == NoUnits ? collect(coords) :
        ustrip_vec.(u"Å", from_device(coords))   # returns SVector{D, Float64/32}
end

# In-place coordinate stripping into a pre-allocated buffer — zero allocations.
function strip_coords_into!(out::AbstractVector{SVector{D,TF}},
                              coords::AbstractVector{SVector{D,T}}) where {D, TF, T}
    if unit(first(coords)[1]) == NoUnits
        @inbounds for i in eachindex(coords)
            out[i] = SVector{D,TF}(coords[i])
        end
    else
        @inbounds for i in eachindex(coords)
            out[i] = SVector{D,TF}(ustrip.(u"Å", coords[i]))
        end
    end
end

# strip_boundary (Cubic + Triclinic) is defined in core Molly (src/interactions/ml_potentials.jl)
# and shared by the CPU and GPU AEV paths — use Molly.strip_boundary.

# ============================================================================
# AEV computation — zero-allocation in-place implementation
# ============================================================================

# The smooth cosine cutoff f_C ([ANI-1] Eq. 2) is defined once in core Molly as
# `cosine_cutoff` (imported via `using Molly`) and shared by the CPU and GPU AEV paths.

# Replace the d-th component of SVector sv with value x.
@inline function setcomp(sv::SVector{D,T}, d::Int, x::T) where {D,T}
    SVector{D,T}(ntuple(k -> k == d ? x : sv[k], Val(D)))
end

# Radial AEV — radial symmetry function G^R, [ANI-1] Eq. (3):
#   G^R_{η,R_s} = Σ_{j≠i} exp(−η·(R_ij − R_s)²) · f_C(R_ij)
# One element per (η_R, r_s_R) pair, computed separately per neighbour species (the
# `base` offset selects the species block). ANI-2x uses one η_R and 16 shifts R_s.
# The 0.25 prefactor follows the TorchANI reference implementation (not in the paper).
# Writes into G (a pre-allocated view of the AEV matrix row);
# nbr_coords/nbr_species hold the first n_nbr neighbours.
function radial_aev!(G::AbstractVector{T}, coord_i::SVector{D,T},
                      nbr_coords, nbr_species, n_nbr::Int,
                      η_R, r_s_R, r_c_R::T, n_species::Int) where {D,T}
    fill!(G, zero(T))
    n_eta = length(η_R)
    n_shf = length(r_s_R)
    for idx in 1:n_nbr
        dr = nbr_coords[idx] - coord_i
        r  = T(norm(dr))
        r >= r_c_R && continue
        fc   = cosine_cutoff(r, r_c_R)
        sj   = nbr_species[idx]
        base = (sj - 1) * n_eta * n_shf
        for ki in 1:n_eta
            for kj in 1:n_shf
                @inbounds G[base + (ki-1)*n_shf + kj] +=
                    T(0.25) * exp(-η_R[ki] * (r - r_s_R[kj])^2) * fc
            end
        end
    end
end

# Angular AEV — modified angular symmetry function G^A, [ANI-1] Eq. (4):
#   G^A_{η,ζ,R_s,θ_s} = 2^{1−ζ} · Σ_{j,k≠i} (1 + cos(θ_ijk − θ_s))^ζ
#                       · exp(−η·((R_ij + R_ik)/2 − R_s)²) · f_C(R_ij) · f_C(R_ik)
# Sum over unordered pairs of neighbours (j,k). One element per (η_A, r_s_A, θ_s) and
# per unordered species pair (the `pair_idx`/`base` offset). prefac0 = 2^{1−ζ},
# r_avg = (R_ij+R_ik)/2, fc_jk = f_C(R_ij)·f_C(R_ik), theta = θ_ijk.
# θ is taken as acos(0.95·cosθ): the 0.95 is the TorchANI NaN guard against |cosθ|→1.
# Phase 1 caches r/f_C/Δr per neighbour; phase 2 loops valid pairs. Writes into G;
# uses pre-allocated scratch vectors rj_buf, fcj_buf, drj_buf, ok_buf (sized ≥ n_nbr).
function angular_aev!(G::AbstractVector{T}, coord_i::SVector{D,T},
                       nbr_coords, nbr_species, n_nbr::Int,
                       η_A, r_s_A, θ_s, ζ::T, r_c_A::T, n_species::Int,
                       rj_buf, fcj_buf, drj_buf, ok_buf) where {D,T}
    fill!(G, zero(T))
    n_eta   = length(η_A)
    n_shf_r = length(r_s_A)
    n_th    = length(θ_s)
    prefac0 = T(2)^(one(T) - ζ)

    for idx in 1:n_nbr
        d = nbr_coords[idx] - coord_i
        r = T(norm(d))
        ok_buf[idx]  = r < r_c_A
        rj_buf[idx]  = r
        fcj_buf[idx] = ok_buf[idx] ? cosine_cutoff(r, r_c_A) : zero(T)
        drj_buf[idx] = d
    end

    for j in 1:n_nbr
        ok_buf[j] || continue
        sj = nbr_species[j]
        for k in (j+1):n_nbr
            ok_buf[k] || continue
            sk = nbr_species[k]
            s1, s2 = sj <= sk ? (sj, sk) : (sk, sj)
            pair_idx = (s1-1)*n_species - (s1-1)*(s1-2)÷2 + (s2-s1+1)

            r_avg  = (rj_buf[j] + rj_buf[k]) * T(0.5)
            fc_jk  = fcj_buf[j] * fcj_buf[k]
            cos_th = clamp(dot(drj_buf[j], drj_buf[k]) / (rj_buf[j] * rj_buf[k]), T(-1), T(1))
            theta  = acos(T(0.95) * cos_th)   # 0.95 matches TorchANI NaN guard

            base = (pair_idx-1) * n_eta * n_shf_r * n_th
            for p in 1:n_eta
                for q in 1:n_shf_r
                    r_factor = prefac0 * fc_jk * exp(-η_A[p] * (r_avg - r_s_A[q])^2)
                    for l in 1:n_th
                        ang = (one(T) + cos(theta - θ_s[l]))^ζ
                        @inbounds G[base + ((p-1)*n_shf_r + (q-1))*n_th + l] += r_factor * ang
                    end
                end
            end
        end
    end
end

# Per-chunk scratch for the threaded AEV loop. Each chunk (≈ one thread) gets its
# own scratch so the central-atom loop has no shared mutable state to race on.
struct AEVScratch{D, T}
    nbr_coords  :: Vector{SVector{D, T}}
    nbr_species :: Vector{Int}
    rj          :: Vector{T}
    fcj         :: Vector{T}
    drj         :: Vector{SVector{D, T}}
    ok          :: Vector{Bool}
end

function AEVScratch{D, T}(max_nbrs::Int) where {D, T}
    AEVScratch{D, T}(
        Vector{SVector{D, T}}(undef, max_nbrs),
        Vector{Int}(undef, max_nbrs),
        Vector{T}(undef, max_nbrs),
        Vector{T}(undef, max_nbrs),
        Vector{SVector{D, T}}(undef, max_nbrs),
        Vector{Bool}(undef, max_nbrs),
    )
end

# Internal buffer struct (lazily allocated per ANIPotential, sized for a given system).
mutable struct AEVBuffers{D, T}
    n_atoms     :: Int
    aevs        :: Matrix{T}                  # (n_atoms, aev_len) — AEV output
    nn_batch    :: Matrix{Float32}            # (aev_len, n_atoms) — reusable per-species NN input
    scratch     :: Vector{AEVScratch{D, T}}   # one scratch set per chunk/thread (no shared races)
    species_idx  :: Vector{Int}               # per-atom species indices (reused across calls)
    idx_to_elem  :: Dict{Int, String}         # inverse of species_map (precomputed once)
    coords_strip :: Vector{SVector{D,T}}      # pre-allocated strip buffer (avoids strip_coords alloc)
    group_atoms  :: Vector{Vector{Int}}       # group_atoms[s] = atom indices of species s (NN batching)
    group_count  :: Vector{Int}               # group_count[s] = #atoms of species s this call
    # CSR per-atom adjacency built once per call from the passed-in NeighborList:
    # atom i's neighbours are nbr_idx[nbr_off[i]+1 : nbr_off[i+1]] (half-pairs symmetrised).
    nbr_off      :: Vector{Int32}             # (n_atoms+1,) CSR offsets
    nbr_idx      :: Vector{Int32}             # (2·total_pairs,) flattened neighbour indices (grows)
    nbr_cursor   :: Vector{Int32}             # (n_atoms,) scatter cursor scratch
end

# Return (lazily allocated) AEVBuffers for this potential + system size.
# Reallocates if n_atoms or T changes (e.g. first call on a new system size).
function get_aev_buf(inter::ANIPotential, n_atoms::Int, ::Val{D}, ::Type{T}) where {D, T}
    buf = inter.buffers[]
    if buf === nothing || buf.n_atoms < n_atoms || eltype(buf.aevs) != T
        p = inter.aev_params
        n_species = length(inter.species_map)
        n_rad_per = length(p.η_R) * length(p.r_s_R)
        n_ang_per = length(p.η_A) * length(p.r_s_A) * length(p.θ_s)
        n_pairs   = n_species * (n_species + 1) ÷ 2
        aev_len   = n_species * n_rad_per + n_pairs * n_ang_per
        max_nbrs  = n_atoms
        idx_to_elem = Dict{Int,String}(v => k for (k, v) in inter.species_map)
        # One scratch set per thread so the central-atom loop can run with
        # Threads.@threads :static and no shared mutable state.
        nchunks = max(1, Threads.nthreads())
        scratch = AEVScratch{D, T}[AEVScratch{D, T}(max_nbrs) for _ in 1:nchunks]
        # Per-species index buckets for batched NN evaluation (one Lux.apply per element).
        group_atoms = Vector{Int}[Vector{Int}(undef, n_atoms) for _ in 1:n_species]
        buf = AEVBuffers{D, T}(
            n_atoms,
            zeros(T, n_atoms, aev_len),
            zeros(Float32, aev_len, n_atoms),
            scratch,
            Vector{Int}(undef, n_atoms),
            idx_to_elem,
            Vector{SVector{D,T}}(undef, n_atoms),
            group_atoms,
            zeros(Int, n_species),
            Vector{Int32}(undef, n_atoms + 1),
            Int32[],                       # nbr_idx grows to 2·total_pairs on first neighbour call
            Vector{Int32}(undef, n_atoms),
        )
        inter.buffers[] = buf
    end
    return buf::AEVBuffers{D, T}
end

# Compute the AEV rows for one contiguous chunk of central atoms, using a single
# private scratch set. Each row = radial block ([ANI-1] Eq. 3, via radial_aev!)
# followed by angular block ([ANI-1] Eq. 4, via angular_aev!). Output rows
# buf.aevs[atom_i, :] are disjoint per atom, so distinct chunks never touch the
# same memory — safe to run on separate threads.
function aev_chunk!(buf::AEVBuffers{D,T}, atom_range, sc::AEVScratch{D,T},
                     coords, species_indices, use_nl::Bool, boundary, p, n_species::Int,
                     r_c_R::T, r_c_A::T, r_c_max::T, split::Int, aev_len::Int) where {D,T}
    n_atoms = length(coords)
    for atom_i in atom_range
        # Build neighbor list into this chunk's private scratch (no push!, no heap alloc).
        n_nbrs = 0
        if !use_nl
            # No neighbour finder: O(N) all-pairs scan. Use vector() so the displacement
            # respects the minimum-image convention under periodic boundaries; store the
            # imaged neighbour position so radial_aev!/angular_aev! (which subtract
            # coord_i) see the correct separation.
            for j in 1:n_atoms
                j == atom_i && continue
                dr = vector(coords[atom_i], coords[j], boundary)
                norm(dr) < r_c_max || continue
                n_nbrs += 1
                sc.nbr_coords[n_nbrs]  = coords[atom_i] + dr
                sc.nbr_species[n_nbrs] = Int(species_indices[j])
            end
        else
            # Read only atom_i's slice of the CSR adjacency built once from the passed-in
            # NeighborList — O(k), not a rescan of the whole list.
            @inbounds for jj in (buf.nbr_off[atom_i] + 1):buf.nbr_off[atom_i + 1]
                nbr_idx = Int(buf.nbr_idx[jj])
                dr = vector(coords[atom_i], coords[nbr_idx], boundary)
                n_nbrs += 1
                sc.nbr_coords[n_nbrs]  = coords[atom_i] + dr
                sc.nbr_species[n_nbrs] = Int(species_indices[nbr_idx])
            end
        end

        # Write AEV components directly into pre-allocated output rows.
        radial_aev!(@view(buf.aevs[atom_i, 1:split]),
                     coords[atom_i], sc.nbr_coords, sc.nbr_species, n_nbrs,
                     p.η_R, p.r_s_R, r_c_R, n_species)
        angular_aev!(@view(buf.aevs[atom_i, split+1:aev_len]),
                      coords[atom_i], sc.nbr_coords, sc.nbr_species, n_nbrs,
                      p.η_A, p.r_s_A, p.θ_s, T(p.ζ), r_c_A, n_species,
                      sc.rj, sc.fcj, sc.drj, sc.ok)
    end
    return nothing
end

# Fill a CSR per-atom adjacency (off, idx) from a NeighborList in a single counting-sort
# pass. Each half-pair (i,j) contributes j to atom i and i to atom j (the NeighborList
# stores each pair once). `cur` is an n_atoms scatter-cursor scratch. O(total_pairs).
function fill_csr!(off, cur, idx, neighbors, n_atoms::Int, npairs::Int)
    @inbounds for a in 1:(n_atoms + 1)
        off[a] = zero(Int32)
    end
    # Pass 1: per-atom degree, accumulated into off[a+1].
    @inbounds for ni in 1:npairs
        i, j, _ = neighbors[ni]
        off[Int(i) + 1] += one(Int32)
        off[Int(j) + 1] += one(Int32)
    end
    # Prefix sum → offsets (atom a's slice is idx[off[a]+1 : off[a+1]]).
    @inbounds for a in 1:n_atoms
        off[a + 1] += off[a]
    end
    # Pass 2: scatter both directions of each half-pair.
    @inbounds for a in 1:n_atoms
        cur[a] = off[a]
    end
    @inbounds for ni in 1:npairs
        i, j, _ = neighbors[ni]
        ii = Int(i); jj = Int(j)
        cur[ii] += one(Int32); idx[cur[ii]] = Int32(jj)
        cur[jj] += one(Int32); idx[cur[jj]] = Int32(ii)
    end
    return nothing
end

# Buffered variant: build the CSR into the reusable AEVBuffers arrays (grows nbr_idx to
# fit, zero-alloc after warmup). Replaces the previous O(N × total_pairs) per-atom rescan.
function neighbors_to_csr!(buf::AEVBuffers, neighbors, n_atoms::Int)
    npairs = length(neighbors)
    total  = 2 * npairs
    length(buf.nbr_idx) < total && resize!(buf.nbr_idx, total)
    fill_csr!(buf.nbr_off, buf.nbr_cursor, buf.nbr_idx, neighbors, n_atoms, npairs)
    return nothing
end

# Compute AEVs for all atoms, writing into buf.aevs (zero allocations after buf is warm).
# Returns a view of buf.aevs — callers must not hold onto it across calls.
# When a NeighborList is passed in, its per-atom adjacency is built once (CSR) and each
# chunk reads only its atoms' slices. Partitions the central-atom loop into
# `length(buf.scratch)` contiguous chunks with Threads.@threads :static (chunk c → thread
# c → scratch c). With one thread it is a plain serial loop (bit-identical, zero allocs).
function compute_aevs_buf!(buf::AEVBuffers{D,T},
                             coords::AbstractVector{SVector{D,T}},
                             species_indices::AbstractVector{<:Integer},
                             neighbors,
                             boundary,
                             p,
                             n_species::Int) where {D,T}
    n_atoms = length(coords)
    r_c_R   = T(p.r_c_R)
    r_c_A   = T(p.r_c_A)
    r_c_max = max(r_c_R, r_c_A)
    n_rad_per = length(p.η_R) * length(p.r_s_R)
    n_pairs   = n_species * (n_species + 1) ÷ 2
    n_ang_per = length(p.η_A) * length(p.r_s_A) * length(p.θ_s)
    aev_len   = n_species * n_rad_per + n_pairs * n_ang_per
    split     = n_species * n_rad_per

    use_nl = !isnothing(neighbors)
    use_nl && neighbors_to_csr!(buf, neighbors, n_atoms)

    nchunks = clamp(length(buf.scratch), 1, max(n_atoms, 1))
    if nchunks == 1
        aev_chunk!(buf, 1:n_atoms, buf.scratch[1], coords, species_indices,
                    use_nl, boundary, p, n_species, r_c_R, r_c_A, r_c_max, split, aev_len)
    else
        base = n_atoms ÷ nchunks
        rem  = n_atoms % nchunks
        # `:static` fixes the chunk→iteration mapping so chunk c always uses scratch[c]
        # (its own buffer, no sharing). It is not about load balancing — the chunks are
        # equal-sized and nchunks == length(buf.scratch) == nthreads, so one chunk runs per
        # thread. (Plain @threads would also be correct since the scratch is per-chunk, but
        # :static avoids the dynamic-scheduler overhead for this fixed, uniform partition.)
        Threads.@threads :static for c in 1:nchunks
            lo = (c - 1) * base + min(c - 1, rem) + 1
            hi = c * base + min(c, rem)
            lo <= hi && aev_chunk!(buf, lo:hi, buf.scratch[c], coords, species_indices,
                                    use_nl, boundary, p, n_species, r_c_R, r_c_A, r_c_max,
                                    split, aev_len)
        end
    end
    return @view buf.aevs[1:n_atoms, :]
end

# Public interface: allocating version (used by potential_energy and for standalone use).
# Same AEV equations as the buffered path: radial block = [ANI-1] Eq. 3 (G^R),
# angular block = [ANI-1] Eq. 4 (G^A), both using the f_C cutoff of [ANI-1] Eq. 2.
function Molly.compute_aevs(coords::AbstractVector{SVector{D,T}},
                             species_indices::AbstractVector{<:Integer},
                             neighbors,
                             boundary,
                             p,
                             n_species::Int) where {D,T}
    n_atoms   = length(coords)
    n_rad_per = length(p.η_R) * length(p.r_s_R)
    n_ang_per = length(p.η_A) * length(p.r_s_A) * length(p.θ_s)
    n_pairs   = n_species * (n_species + 1) ÷ 2
    aev_len   = n_species * n_rad_per + n_pairs * n_ang_per
    split     = n_species * n_rad_per
    r_c_R     = T(p.r_c_R)
    r_c_A     = T(p.r_c_A)
    r_c_max   = max(r_c_R, r_c_A)

    aevs        = zeros(T, n_atoms, aev_len)
    nbr_coords  = Vector{SVector{D,T}}(undef, n_atoms)
    nbr_species = Vector{Int}(undef, n_atoms)
    rj_buf      = Vector{T}(undef, n_atoms)
    fcj_buf     = Vector{T}(undef, n_atoms)
    drj_buf     = Vector{SVector{D,T}}(undef, n_atoms)
    ok_buf      = Vector{Bool}(undef, n_atoms)

    # Build the per-atom CSR adjacency once from the passed-in NeighborList (integer-only)
    # so each atom reads its own slice rather than rescanning the list.
    use_nl = !isnothing(neighbors)
    csr_off = Vector{Int32}(undef, use_nl ? n_atoms + 1 : 0)
    csr_idx = Vector{Int32}(undef, use_nl ? 2 * length(neighbors) : 0)
    if use_nl
        fill_csr!(csr_off, Vector{Int32}(undef, n_atoms), csr_idx,
                   neighbors, n_atoms, length(neighbors))
    end

    for atom_i in 1:n_atoms
        n_nbrs = 0
        if !use_nl
            # All-pairs scan with minimum-image displacement (see aev_chunk!).
            for j in 1:n_atoms
                j == atom_i && continue
                dr = vector(coords[atom_i], coords[j], boundary)
                norm(dr) < r_c_max || continue
                n_nbrs += 1
                nbr_coords[n_nbrs]  = coords[atom_i] + dr
                nbr_species[n_nbrs] = Int(species_indices[j])
            end
        else
            for jj in (csr_off[atom_i] + 1):csr_off[atom_i + 1]
                nbr_idx = Int(csr_idx[jj])
                dr = vector(coords[atom_i], coords[nbr_idx], boundary)
                n_nbrs += 1
                nbr_coords[n_nbrs]  = coords[atom_i] + dr
                nbr_species[n_nbrs] = Int(species_indices[nbr_idx])
            end
        end
        radial_aev!(@view(aevs[atom_i, 1:split]),
                     coords[atom_i], nbr_coords, nbr_species, n_nbrs,
                     p.η_R, p.r_s_R, r_c_R, n_species)
        angular_aev!(@view(aevs[atom_i, split+1:aev_len]),
                      coords[atom_i], nbr_coords, nbr_species, n_nbrs,
                      p.η_A, p.r_s_A, p.θ_s, T(p.ζ), r_c_A, n_species,
                      rj_buf, fcj_buf, drj_buf, ok_buf)
    end
    return aevs
end

# ============================================================================
# HDF5 weight loader
# ============================================================================

# The element networks use the public `Molly.celu01` activation (imported via `using Molly`),
# a single shared definition also differentiated analytically by the on-device backward.

# Build one element's Lux.Chain from the HDF5 group at /ensemble_idx/element/.
# Dense layer indices 0, 2, 4, ... in the group; activations implicit between them.
function build_element_model(elem_grp::HDF5.Group)
    layer_ids = sort(parse.(Int, keys(elem_grp)))   # [0, 2, 4, 6]
    n_dense   = length(layer_ids)
    lux_layers = []
    ps_pairs   = Pair{Symbol,NamedTuple}[]

    for (k, li) in enumerate(layer_ids)
        lg  = elem_grp[string(li)]
        # HDF5.jl reads arrays transposed relative to Python (column-major vs row-major).
        # Python stored weight as (out, in); Julia reads it as (in, out) → permute back.
        W   = permutedims(Float32.(read(lg["weight"])))   # (out, in)
        b   = Float32.(read(lg["bias"]))                  # (out,)
        out_sz, in_sz = size(W)
        act = k < n_dense ? celu01 : identity
        push!(lux_layers, Lux.Dense(in_sz => out_sz, act))
        push!(ps_pairs, Symbol("layer_", k) => (weight = W, bias = b))
    end

    model = Lux.Chain(lux_layers...)
    ps    = NamedTuple(ps_pairs)
    # We only need the model's state `st`; the randomly-initialised parameters from
    # Lux.setup are discarded and replaced by `ps` loaded from HDF5, so the RNG seed is
    # irrelevant to the result — a fixed Xoshiro(0) just makes construction deterministic.
    _, st = Lux.setup(Xoshiro(0), model)
    return model, ps, st
end

# Read an HDF5 dataset as a Vector{T}, handling scalar datasets gracefully.
h5vec(grp, name, T) = T.(vcat(read(grp[name])))

function Molly.ANIPotential(path::String;
                           force_units  = u"eV/Å",
                           energy_units = u"eV",
                           T            = Float32,
                           ensemble_idx = nothing)
    h5open(path, "r") do h5
        # AEV hyperparameters
        ag    = h5["aev_params"]
        r_c_R = T(only(read(ag["Rcr"])))
        r_c_A = T(only(read(ag["Rca"])))
        η_R   = h5vec(ag, "EtaR", T)
        r_s_R = h5vec(ag, "ShfR", T)
        η_A   = h5vec(ag, "EtaA", T)
        # TorchANI naming is counterintuitive: ShfA holds *radial* shifts (Å) and
        # ShfZ holds *angular* shifts (rad) for the angular AEV Gaussian.
        r_s_A = h5vec(ag, "ShfA", T)   # ShfA = radial shifts for angular Gaussian (Å)
        θ_s   = h5vec(ag, "ShfZ", T)   # ShfZ = angular shifts θ_s (rad)
        ζ     = T(only(read(ag["Zeta"])))
        species_list = String.(read(ag["species"]))

        aev_params = (η_R=η_R, r_s_R=r_s_R, r_c_R=r_c_R,
                      η_A=η_A, r_s_A=r_s_A, θ_s=θ_s, ζ=ζ, r_c_A=r_c_A)

        species_map   = Dict{String,Int}(s => i for (i, s) in enumerate(species_list))
        self_energies = h5vec(ag, "self_energies", T)   # (n_species,) Hartree

        # Determine which ensemble members to load.
        # All integer-keyed top-level groups are ensemble members.
        ens_indices = if isnothing(ensemble_idx)
            sort([parse(Int, k) for k in keys(h5) if tryparse(Int, k) !== nothing])
        else
            [ensemble_idx]
        end

        # Build shared model architecture from the first ensemble member.
        syms = Tuple(Symbol.(species_list))
        first_grp = h5[string(first(ens_indices))]
        models_v  = []
        for elem in species_list
            m, _, _ = build_element_model(first_grp[elem])
            push!(models_v, m)
        end
        model_nt = NamedTuple{syms}(Tuple(models_v))

        # Load parameters and states for each ensemble member.
        ps_list = NamedTuple[]
        st_list = NamedTuple[]
        for idx in ens_indices
            ens_grp = h5[string(idx)]
            ps_v = []; st_v = []
            for elem in species_list
                _, ps, st = build_element_model(ens_grp[elem])
                push!(ps_v, ps); push!(st_v, st)
            end
            push!(ps_list, NamedTuple{syms}(Tuple(ps_v)))
            push!(st_list, NamedTuple{syms}(Tuple(st_v)))
        end

        cutoff = T(max(r_c_R, r_c_A))
        return ANIPotential(model_nt, ps_list, st_list, species_map,
                            aev_params, self_energies, cutoff, force_units, energy_units,
                            Ref{Any}(nothing))
    end
end

# ============================================================================
# AtomsCalculators interface
# ============================================================================

# Bucket atoms by species into buf.group_atoms / buf.group_count (no heap alloc after warmup).
# Independent of ensemble member, so computed once per energy call.
function bucket_species!(group_atoms, group_count, species_idx)
    fill!(group_count, 0)
    @inbounds for atom_i in eachindex(species_idx)
        s = species_idx[atom_i]
        c = (group_count[s] += 1)
        group_atoms[s][c] = atom_i
    end
    return nothing
end

# Internal: energy (Hartree) for one ensemble member given pre-computed AEVs.
# Implements the per-atom energy sum E = Σ_i E_i ([ANI-1] Eq. 1): each atom's AEV is
# mapped by its element network to E_i (plus an additive self-energy reference shift).
# Batched by species: one Lux.apply per element (an (aev_len, n_s) matmul) instead of
# one tiny matmul per atom. nn_batch is a reusable (aev_len, n_atoms) Float32 scratch.
function ani_energy_single(aevs, idx_to_elem, model, self_energies, ps, st,
                            nn_batch::Matrix{Float32}, group_atoms, group_count, ::Type{T}) where T
    E = zero(T)
    n_species = length(group_count)
    for s in 1:n_species
        ns = group_count[s]
        ns == 0 && continue
        sym = Symbol(idx_to_elem[s])
        # Gather this species' AEV rows into the first ns columns of nn_batch.
        atoms_s = group_atoms[s]
        @inbounds for k in 1:ns
            @views nn_batch[:, k] .= aevs[atoms_s[k], :]
        end
        batch  = @view nn_batch[:, 1:ns]
        out, _ = Lux.apply(getfield(model, sym), batch, getfield(ps, sym), getfield(st, sym))
        # out is (1, ns); add per-atom NN output plus this species' self-energy.
        s_e = self_energies[s]
        @inbounds for k in 1:ns
            E += T(out[1, k]) + s_e
        end
    end
    return E
end

# Internal: compute raw energy (Hartree) averaged over all ensemble members.
# Total energy E = Σ_i E_i ([ANI-1] Eq. 1) from each member, then the ensemble
# average E = (1/M) Σ_m E^(m) ([ANI-1x] Smith et al., J. Chem. Phys. 2018, 148,
# 241733 — ANI-2x uses M = 8 networks).
# Uses the lazy AEVBuffers cache in inter.buffers for zero allocations after warmup.
function ani_raw_energy(coords_strip::AbstractVector{SVector{D,T}},
                         species_idx::AbstractVector{Int},
                         boundary, inter::ANIPotential,
                         neighbors) where {D,T}
    n_species = length(inter.species_map)
    bdy_strip = Molly.strip_boundary(boundary)
    n_atoms   = length(coords_strip)

    # Use cached buffers (lazily allocated on first call, 0 allocs on subsequent).
    buf  = get_aev_buf(inter, n_atoms, Val(D), T)
    aevs = compute_aevs_buf!(buf, coords_strip, species_idx, neighbors, bdy_strip,
                               inter.aev_params, n_species)

    # Bucket atoms by species once (shared across ensemble members).
    bucket_species!(buf.group_atoms, buf.group_count, species_idx)

    n_ens = length(inter.ps_vec)
    E = zero(T)
    for i in 1:n_ens
        E += ani_energy_single(aevs, buf.idx_to_elem, inter.model,
                                inter.self_energies, inter.ps_vec[i], inter.st_vec[i],
                                buf.nn_batch, buf.group_atoms, buf.group_count, T)
    end
    return E / n_ens
end

function AtomsCalculators.potential_energy(sys::System{D, AT, T},
                                           inter::ANIPotential;
                                           kwargs...) where {D, AT, T}
    n_atoms = length(sys.coords)
    nbrs    = get(kwargs, :neighbors, nothing)

    # Use cached buffers — all pre-allocated, zero heap allocations after first call.
    buf = get_aev_buf(inter, n_atoms, Val(D), T)
    strip_coords_into!(buf.coords_strip, sys.coords)
    @inbounds for i in 1:n_atoms
        buf.species_idx[i] = inter.species_map[sys.atoms_data[i].element]
    end
    coords_strip = @view buf.coords_strip[1:n_atoms]
    species_idx  = @view buf.species_idx[1:n_atoms]

    E_ha = ani_raw_energy(coords_strip, species_idx, sys.boundary, inter, nbrs)
    return Molly.ani_energy_to_units(E_ha * T(Molly.HARTREE_TO_EV), sys.energy_units)
end

# ============================================================================
# GPU-portable ANI kernels: AEV, on-device energy (compute_ani_energy_ka), the analytic
# forces (compute_ani_forces_ka), and the single AtomsCalculators.forces!. KernelAbstractions
# is a strong Molly dependency, so these run whenever this (Lux + HDF5) extension is loaded.
# ============================================================================

# KernelAbstractions-based AEV computation for ANI ML potentials.
# Active whenever Lux and HDF5 are loaded (KernelAbstractions is a strong Molly dependency).
# Provides GPU-portable @kernel implementations of the radial/angular AEV so the same
# code runs on CPU (via KA CPU backend) and CUDA/ROCm/Metal GPUs.
#
# BOUNDARY NOTE: these kernels compute neighbour displacements with `Molly.vector(ci,
# coords[j], boundary)`, so they apply the minimum-image convention. Both `CubicBoundary`
# and `TriclinicBoundary` work (the boundary is only ever touched via `vector`). Pass the
# system boundary via the `boundary` kwarg of compute_aevs_ka/compute_ani_energy_ka; it is
# unit-stripped and converted to the coord element type. Omitting it (nothing) uses an
# infinite box, i.e. no minimum image (non-periodic / gas-phase).



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
    coords = coords isa Array ? coords : Array(coords)   # this O(N²) host build can't index a GPU array
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

# Device-resident NN parameter cache. Uploading the per-element weights to the GPU dominates the
# small-N Metal cost, and the weights never change, so cache (ps_dev, st_dev) per potential and
# device type. A trajectory or repeated call then reuses the on-device weights — no re-upload.
const ANI_NN_DEV_CACHE = IdDict{Any, Dict{Any, Any}}()
function ani_nn_dev_params(pot, dev)
    sub = get!(() -> Dict{Any, Any}(), ANI_NN_DEV_CACHE, pot)
    get!(() -> (map(dev, pot.ps_vec), map(dev, pot.st_vec)), sub, typeof(dev))
end

# ============================================================================
# End-to-end on-device ANI energy: GPU AEV + on-device element networks.
# The AEV matrix stays on the compute backend; the per-element Lux networks run on the
# same device (params cached on-device, see ani_nn_dev_params), so a full energy evaluation
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
    ps_dev_all, st_dev_all = ani_nn_dev_params(pot, dev)             # cached on-device weights
    E = 0.0
    for ens_i in 1:n_ens
        for s in 1:n_species
            g = groups[s]
            isempty(g) && continue
            sym   = Symbol(idx_to_elem[s])
            batch = permutedims(aevs[idx_dev[s], :])                  # (aev_len, n_s) on device
            ps_d  = getfield(ps_dev_all[ens_i], sym)
            st_d  = getfield(st_dev_all[ens_i], sym)
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
    ps_dev_all, _ = ani_nn_dev_params(pot, dev)               # cached on-device weights
    E = 0.0
    for ens_i in 1:n_ens
        for s in 1:n_species
            g = groups[s]; isempty(g) && continue
            sym   = Symbol(idx_to_elem[s])
            batch = permutedims(aevs[idx_dev[s], :])          # (aev_len, n_s)
            ps_d  = getfield(ps_dev_all[ens_i], sym)
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
    else   # GPU system: build the unit-carrying increment on the host as concrete SVectors matching
        # the force buffer's element type (ani_force_to_units(..., Array) yields a plain Vector, so
        # rebuild an inline SVector explicitly), then add on-device.
        FU  = eltype(eltype(fs))                       # unit-carrying scalar type of `fs`
        inc = Vector{SVector{D, FU}}(undef, length(F))
        @inbounds for i in eachindex(F)
            fui = Molly.ani_force_to_units(SVector{D, T}(F[i]), sys.force_units, Array)
            inc[i] = SVector{D, FU}(ntuple(k -> fui[k], D))
        end
        fs .+= Molly.to_device(inc, AT)
    end
    return fs
end

end # module MollyLuxExt
