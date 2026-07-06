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
#   _radial_aev!    → [ANI-1] Eq. (3)  radial symmetry function  G^R
#   _angular_aev!   → [ANI-1] Eq. (4)  angular symmetry function G^A
#   _ani_energy_single / _ani_raw_energy → [ANI-1] Eq. (1) energy sum + ensemble average
# The element NN architecture is in _build_element_model; per-atom self-energies are an
# additive reference shift (Hartree). Forces are −∇E (Enzyme reverse-mode AD).

module MollyLuxExt

using Molly
using Molly: from_device, to_device, vector
import AtomsCalculators
using Lux, HDF5
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
function _strip_coords_into!(out::AbstractVector{SVector{D,TF}},
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

# Strip length units from a CubicBoundary so it is consistent with unit-stripped coords (Å).
function strip_boundary(b::CubicBoundary)
    unit(b.side_lengths[1]) == NoUnits ? b :
        CubicBoundary(ustrip.(u"Å", b.side_lengths))
end

# ============================================================================
# AEV computation — zero-allocation in-place implementation
# ============================================================================

# The smooth cosine cutoff f_C ([ANI-1] Eq. 2) is defined once in core Molly as
# `cosine_cutoff` (imported via `using Molly`) and shared by the CPU and GPU AEV paths.

# Replace the d-th component of SVector sv with value x.
@inline function _setcomp(sv::SVector{D,T}, d::Int, x::T) where {D,T}
    SVector{D,T}(ntuple(k -> k == d ? x : sv[k], Val(D)))
end

# Radial AEV — radial symmetry function G^R, [ANI-1] Eq. (3):
#   G^R_{η,R_s} = Σ_{j≠i} exp(−η·(R_ij − R_s)²) · f_C(R_ij)
# One element per (η_R, r_s_R) pair, computed separately per neighbour species (the
# `base` offset selects the species block). ANI-2x uses one η_R and 16 shifts R_s.
# The 0.25 prefactor follows the TorchANI reference implementation (not in the paper).
# Writes into G (a pre-allocated view of the AEV matrix row);
# nbr_coords/nbr_species hold the first n_nbr neighbours.
function _radial_aev!(G::AbstractVector{T}, coord_i::SVector{D,T},
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
function _angular_aev!(G::AbstractVector{T}, coord_i::SVector{D,T},
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
function _get_aev_buf(inter::ANIPotential, n_atoms::Int, ::Val{D}, ::Type{T}) where {D, T}
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
# private scratch set. Each row = radial block ([ANI-1] Eq. 3, via _radial_aev!)
# followed by angular block ([ANI-1] Eq. 4, via _angular_aev!). Output rows
# buf.aevs[atom_i, :] are disjoint per atom, so distinct chunks never touch the
# same memory — safe to run on separate threads.
function _aev_chunk!(buf::AEVBuffers{D,T}, atom_range, sc::AEVScratch{D,T},
                     coords, species_indices, use_nl::Bool, boundary, p, n_species::Int,
                     r_c_R::T, r_c_A::T, r_c_max::T, split::Int, aev_len::Int) where {D,T}
    n_atoms = length(coords)
    for atom_i in atom_range
        # Build neighbor list into this chunk's private scratch (no push!, no heap alloc).
        n_nbrs = 0
        if !use_nl
            # No neighbour finder: O(N) all-pairs scan. Use vector() so the displacement
            # respects the minimum-image convention under periodic boundaries; store the
            # imaged neighbour position so _radial_aev!/_angular_aev! (which subtract
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
        _radial_aev!(@view(buf.aevs[atom_i, 1:split]),
                     coords[atom_i], sc.nbr_coords, sc.nbr_species, n_nbrs,
                     p.η_R, p.r_s_R, r_c_R, n_species)
        _angular_aev!(@view(buf.aevs[atom_i, split+1:aev_len]),
                      coords[atom_i], sc.nbr_coords, sc.nbr_species, n_nbrs,
                      p.η_A, p.r_s_A, p.θ_s, T(p.ζ), r_c_A, n_species,
                      sc.rj, sc.fcj, sc.drj, sc.ok)
    end
    return nothing
end

# Fill a CSR per-atom adjacency (off, idx) from a NeighborList in a single counting-sort
# pass. Each half-pair (i,j) contributes j to atom i and i to atom j (the NeighborList
# stores each pair once). `cur` is an n_atoms scatter-cursor scratch. O(total_pairs).
# All integer work — Enzyme-inactive, so it is safe inside the AD energy function.
function _fill_csr!(off, cur, idx, neighbors, n_atoms::Int, npairs::Int)
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
function _neighbors_to_csr!(buf::AEVBuffers, neighbors, n_atoms::Int)
    npairs = length(neighbors)
    total  = 2 * npairs
    length(buf.nbr_idx) < total && resize!(buf.nbr_idx, total)
    _fill_csr!(buf.nbr_off, buf.nbr_cursor, buf.nbr_idx, neighbors, n_atoms, npairs)
    return nothing
end

# Compute AEVs for all atoms, writing into buf.aevs (zero allocations after buf is warm).
# Returns a view of buf.aevs — callers must not hold onto it across calls.
# When a NeighborList is passed in, its per-atom adjacency is built once (CSR) and each
# chunk reads only its atoms' slices. Partitions the central-atom loop into
# `length(buf.scratch)` contiguous chunks with Threads.@threads :static (chunk c → thread
# c → scratch c). With one thread it is a plain serial loop (bit-identical, zero allocs).
function _compute_aevs_buf!(buf::AEVBuffers{D,T},
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
    use_nl && _neighbors_to_csr!(buf, neighbors, n_atoms)

    nchunks = clamp(length(buf.scratch), 1, max(n_atoms, 1))
    if nchunks == 1
        _aev_chunk!(buf, 1:n_atoms, buf.scratch[1], coords, species_indices,
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
            lo <= hi && _aev_chunk!(buf, lo:hi, buf.scratch[c], coords, species_indices,
                                    use_nl, boundary, p, n_species, r_c_R, r_c_A, r_c_max,
                                    split, aev_len)
        end
    end
    return @view buf.aevs[1:n_atoms, :]
end

# Public interface: allocating version (for Enzyme AD path and standalone use).
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

    # Build the per-atom CSR adjacency once from the passed-in NeighborList (integer-only,
    # Enzyme-inactive) so each atom reads its own slice rather than rescanning the list.
    use_nl = !isnothing(neighbors)
    csr_off = Vector{Int32}(undef, use_nl ? n_atoms + 1 : 0)
    csr_idx = Vector{Int32}(undef, use_nl ? 2 * length(neighbors) : 0)
    if use_nl
        _fill_csr!(csr_off, Vector{Int32}(undef, n_atoms), csr_idx,
                   neighbors, n_atoms, length(neighbors))
    end

    for atom_i in 1:n_atoms
        n_nbrs = 0
        if !use_nl
            # All-pairs scan with minimum-image displacement (see _aev_chunk!).
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
        _radial_aev!(@view(aevs[atom_i, 1:split]),
                     coords[atom_i], nbr_coords, nbr_species, n_nbrs,
                     p.η_R, p.r_s_R, r_c_R, n_species)
        _angular_aev!(@view(aevs[atom_i, split+1:aev_len]),
                      coords[atom_i], nbr_coords, nbr_species, n_nbrs,
                      p.η_A, p.r_s_A, p.θ_s, T(p.ζ), r_c_A, n_species,
                      rj_buf, fcj_buf, drj_buf, ok_buf)
    end
    return aevs
end

# ============================================================================
# HDF5 weight loader
# ============================================================================

# Use the public Molly.celu01 so AD backends (Enzyme, Zygote) can register rules.
const _celu01 = Molly.celu01

# Build one element's Lux.Chain from the HDF5 group at /ensemble_idx/element/.
# Dense layer indices 0, 2, 4, ... in the group; activations implicit between them.
function _build_element_model(elem_grp::HDF5.Group)
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
        act = k < n_dense ? _celu01 : identity
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
_h5vec(grp, name, T) = T.(vcat(read(grp[name])))

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
        η_R   = _h5vec(ag, "EtaR", T)
        r_s_R = _h5vec(ag, "ShfR", T)
        η_A   = _h5vec(ag, "EtaA", T)
        # TorchANI naming is counterintuitive: ShfA holds *radial* shifts (Å) and
        # ShfZ holds *angular* shifts (rad) for the angular AEV Gaussian.
        r_s_A = _h5vec(ag, "ShfA", T)   # ShfA = radial shifts for angular Gaussian (Å)
        θ_s   = _h5vec(ag, "ShfZ", T)   # ShfZ = angular shifts θ_s (rad)
        ζ     = T(only(read(ag["Zeta"])))
        species_list = String.(read(ag["species"]))

        aev_params = (η_R=η_R, r_s_R=r_s_R, r_c_R=r_c_R,
                      η_A=η_A, r_s_A=r_s_A, θ_s=θ_s, ζ=ζ, r_c_A=r_c_A)

        species_map   = Dict{String,Int}(s => i for (i, s) in enumerate(species_list))
        self_energies = _h5vec(ag, "self_energies", T)   # (n_species,) Hartree

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
            m, _, _ = _build_element_model(first_grp[elem])
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
                _, ps, st = _build_element_model(ens_grp[elem])
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
function _bucket_species!(group_atoms, group_count, species_idx)
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
function _ani_energy_single(aevs, idx_to_elem, model, self_energies, ps, st,
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
function _ani_raw_energy(coords_strip::AbstractVector{SVector{D,T}},
                         species_idx::AbstractVector{Int},
                         boundary, inter::ANIPotential,
                         neighbors) where {D,T}
    n_species = length(inter.species_map)
    bdy_strip = strip_boundary(boundary)
    n_atoms   = length(coords_strip)

    # Use cached buffers (lazily allocated on first call, 0 allocs on subsequent).
    buf  = _get_aev_buf(inter, n_atoms, Val(D), T)
    aevs = _compute_aevs_buf!(buf, coords_strip, species_idx, neighbors, bdy_strip,
                               inter.aev_params, n_species)

    # Bucket atoms by species once (shared across ensemble members).
    _bucket_species!(buf.group_atoms, buf.group_count, species_idx)

    n_ens = length(inter.ps_vec)
    E = zero(T)
    for i in 1:n_ens
        E += _ani_energy_single(aevs, buf.idx_to_elem, inter.model,
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
    buf = _get_aev_buf(inter, n_atoms, Val(D), T)
    _strip_coords_into!(buf.coords_strip, sys.coords)
    @inbounds for i in 1:n_atoms
        buf.species_idx[i] = inter.species_map[sys.atoms_data[i].element]
    end
    coords_strip = @view buf.coords_strip[1:n_atoms]
    species_idx  = @view buf.species_idx[1:n_atoms]

    E_ha = _ani_raw_energy(coords_strip, species_idx, sys.boundary, inter, nbrs)
    return Molly.ani_energy_to_units(E_ha * T(Molly.HARTREE_TO_EV), sys.energy_units)
end

# Forces via finite differences (fallback when Enzyme is not loaded).
function AtomsCalculators.forces!(fs,
                                  sys::System{D, AT, T},
                                  inter::ANIPotential;
                                  kwargs...) where {D, AT, T}
    h = T(1e-4)   # Å — must be T (Float64) for numerical precision

    coords_strip = strip_coords(sys.coords)
    species_idx  = [inter.species_map[ad.element] for ad in sys.atoms_data]
    nbrs = get(kwargs, :neighbors, nothing)

    for atom_i in eachindex(coords_strip)
        fi = zero(SVector{D, T})
        for dim in 1:D
            cp = copy(coords_strip)
            cm = copy(coords_strip)
            cp[atom_i] = _setcomp(cp[atom_i], dim, cp[atom_i][dim] + h)
            cm[atom_i] = _setcomp(cm[atom_i], dim, cm[atom_i][dim] - h)
            Ep = _ani_raw_energy(cp, species_idx, sys.boundary, inter, nbrs)
            Em = _ani_raw_energy(cm, species_idx, sys.boundary, inter, nbrs)
            fi = _setcomp(fi, dim, T(-(Ep - Em) / (2h) * Molly.HARTREE_TO_EV))
        end
        fs[atom_i] += Molly.ani_force_to_units(fi, sys.force_units, AT)
    end
    return fs
end

end # module MollyLuxExt
