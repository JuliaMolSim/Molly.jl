# Calculate collective variables

export
    CalcMinDist,
    CalcMaxDist,
    CalcCMDist,
    CalcSingleDist,
    CalcDist,
    calculate_cv,
    cv_gradient,
    calculate_cv!,
    cv_gradient!,
    CalcRg,
    CalcRMSD,
    CalcTorsion

# Does not account for periodic boundary conditions, assumes appropriate unwrapping
function center_of_mass(coords, atoms)
    com = similar(coords, 1)
    center_of_mass!(coords, atoms, com)
    return only(from_device(com))
end

function center_of_mass!(coords, atoms, com, mass_total_buf=nothing)
    masses = mass.(atoms)
    mtot = sum(masses; dims=1)
    com .= sum(masses .* coords; dims=1) ./ mtot
    mass_total_buf === nothing || (mass_total_buf .= mtot)
    return nothing
end

function calculate_virial(cv, args...; kwargs...) end

function pairwise_displacement_matrix(coords_1::AbstractArray{SVector{D, C}},
                                      coords_2::AbstractArray{SVector{D, C}},
                                      calc_type,
                                      boundary) where {D, C}
    c1_col = reshape(coords_1, length(coords_1), 1)
    c2_row = reshape(coords_2, 1, length(coords_2))
    if calc_type == :closest
        return vector.(c1_col, c2_row, (boundary,))
    else
        return c2_row .- c1_col
    end
end

function pairwise_distance_matrix(coords_1, coords_2, calc_type, boundary)
    return norm.(pairwise_displacement_matrix(coords_1, coords_2, calc_type, boundary))
end

# Finds the pair (i, j) minimizing/maximizing the distance between two groups of atoms, using
# `extremum_fn = findmin`/`findmax`. Returns the indices, the extremal distance, and the
# coords_1[i] -> coords_2[j] displacement vector, without scalar-indexing (safe for CuArray).
#
# CPU / generic fallback: materializes the full group_a x group_b displacement matrix. Fine on
# CPU; on GPU this is replaced by `extremal_pair_fused` below, which avoids the O(Na*Nb) matrix.
function extremal_pair_dense(coords_1, coords_2, calc_type, extremum_fn, boundary)
    diffs = pairwise_displacement_matrix(coords_1, coords_2, calc_type, boundary)
    dist_matrix = norm.(diffs)
    d, idx = extremum_fn(dist_matrix)
    i, j = Tuple(idx)
    r_ij = only(from_device(diffs[i:i, j:j]))
    return i, j, d, r_ij
end

# A fancy-index @view of a CuArray isn't itself an AbstractGPUArray, so unwrap via `parent`.
is_gpu_resident(x::AbstractGPUArray) = true
is_gpu_resident(x::SubArray) = is_gpu_resident(parent(x))
is_gpu_resident(x) = false

function extremal_pair(coords_1, coords_2, calc_type, extremum_fn, boundary)
    if is_gpu_resident(coords_1)
        return extremal_pair_fused(coords_1, coords_2, calc_type, extremum_fn, boundary)
    else
        return extremal_pair_dense(coords_1, coords_2, calc_type, extremum_fn, boundary)
    end
end

# GPU-native `extremal_pair`: avoids materializing the dense group_a x group_b displacement matrix
# (O(group_a * group_b) memory -- OOMs for large groups). Fallback used with no persistent
# `MinMaxScratch` (e.g. ad-hoc `calculate_cv`/`cv_gradient` calls); BiasPotential's usual path uses
# the tile/finalize kernels below instead, which avoid findmin/findmax's host sync via a
# device-side reduction -- see the comment above those kernels for the full design.

"""
    MinMaxScratch

Persistent GPU scratch for the fused CalcMinDist/CalcMaxDist `calculate_cv!`/`cv_gradient!` path
(see `mindist_tile_kernel!` below). `idx1_dev`/`idx2_dev` are device copies of
`cv.atom_inds_1`/`atom_inds_2`, uploaded once and reused, avoiding CPU-GPU transfere and allowing CuGraph. `winner_i`/`winner_j`/`r_ij` cache the most recent winning pair so
`calculate_virial_dist!` can reuse it via `ExtremalPairCache` instead of recomputing the search.
"""
mutable struct MinMaxScratch{IV, DV, JV, RV, SV, R1V}
    idx1_dev::IV
    idx2_dev::IV
    out_dist::DV
    out_i::JV
    out_j::JV
    out_disp::RV
    winner_i::SV
    winner_j::SV
    r_ij::R1V
end

# Caps mindist_tile_kernel!'s parallel workers (also the finalize kernel's serial-scan length,
# trading finalize cost against tile parallelism). 4096 keeps that scan in the few-microsecond
# range while still giving a small/lopsided group (e.g. na=5, nb=1e5) far more concurrency than
# one-thread-per-row would.
const MINDIST_TILE_CAP = 4096

"""
    ExtremalPairCache

Caches the `(i, j, d, r_ij)` result of an `extremal_pair` call made inside `cv_gradient!`, so a
`calculate_virial_dist!` call later in the same timestep can reuse it instead of recomputing the
O(group_a * group_b) search. Populated by `BiasPotential` (bias.jl); otherwise unused.
"""
mutable struct ExtremalPairCache
    valid::Bool
    i::Int
    j::Int
    d::Any
    r_ij::Any
end

@kernel inbounds=true function extremal_pair_row_kernel!(out_dist, out_j, out_disp,
                                                          @Const(coords_1), @Const(coords_2),
                                                          boundary, closest::Bool, ::Val{is_min}) where is_min
    i = @index(Global, Linear)
    if i <= length(coords_1)
        ci = coords_1[i]
        r1 = closest ? vector(ci, coords_2[1], boundary) : coords_2[1] - ci
        best_d, best_j, best_disp = norm(r1), 1, r1
        for j in 2:length(coords_2)
            rij = closest ? vector(ci, coords_2[j], boundary) : coords_2[j] - ci
            d = norm(rij)
            better = is_min ? (d < best_d) : (d > best_d)
            if better
                best_d, best_j, best_disp = d, j, rij
            end
        end
        out_dist[i] = best_d
        out_j[i] = best_j
        out_disp[i] = best_disp
    end
end

function extremal_pair_fused(coords_1, coords_2, calc_type, extremum_fn, boundary)
    na = length(coords_1)
    out_dist = similar(coords_1, eltype(eltype(coords_1)), na)
    out_j = similar(coords_1, Int, na)
    out_disp = similar(coords_1, na)

    closest = calc_type == :closest
    is_min = extremum_fn === findmin
    backend = get_backend(coords_1)
    kernel! = extremal_pair_row_kernel!(backend, min(na, 256))
    kernel!(out_dist, out_j, out_disp, coords_1, coords_2, boundary, closest, Val(is_min); ndrange=na)

    d, i = extremum_fn(out_dist)
    j = only(from_device(out_j[i:i]))
    r_ij = only(from_device(out_disp[i:i]))
    return i, j, d, r_ij
end

# --------------------------------------------------------------
# Fused path for CalcMinDist/CalcMaxDist's calculate_cv!/cv_gradient!, used whenever a persistent
# `MinMaxScratch` is supplied and `coords` is GPU-resident. No atomics, no host syncs.
#
# A one-thread-per-row design (ndrange=group_a) starves when group_a is small (the realistic CV
# case, e.g. na=5 vs a much larger group_b). Fixed by tiling over the *flattened* na*nb pair space:
#  1. `mindist_tile_kernel!` -- T=min(na*nb, MINDIST_TILE_CAP) workers grid-stride over disjoint
#     slices of all na*nb pairs, each keeping a running local winner (O(T) output, not O(na)).
#  2. `mindist_finalize_value_kernel!`/`mindist_finalize_grad_kernel!` -- serial scan over the
#     T (<=MINDIST_TILE_CAP) worker outputs, bounded regardless of group size.
#  3. `mindist_clear_grad_kernel!` -- clears any stale nonzero `grad` entry from a previous call's
#     different winning pair, in parallel; must run before the finalize kernel writes the new
#     winning pair's 2 entries. `calculate_cv!` needs (1)+(2); `cv_gradient!` needs all three.
# `MinMaxScratch.idx1_dev`/`idx2_dev` are GPU-resident copies of `cv.atom_inds_1/2`.

# Worker `tid` (of T) visits pairs `tid`, `tid+T`, `tid+2T`, ... (recovered as (i, j) via div/mod
# on group B's length); T = min(na*nb, MINDIST_TILE_CAP) guarantees every worker gets a pair.
@kernel inbounds=true function mindist_tile_kernel!(out_dist, out_i, out_j, out_disp, @Const(coords),
                                                     @Const(idx1), @Const(idx2), boundary,
                                                     closest::Bool, ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    T = length(out_dist)
    nb = length(idx2)
    total = length(idx1) * nb

    k = tid
    i = (k - 1) ÷ nb + 1
    j = (k - 1) % nb + 1
    ci, cj = coords[idx1[i]], coords[idx2[j]]
    r1 = closest ? vector(ci, cj, boundary) : cj - ci
    best_d, best_i, best_j, best_disp = norm(r1), i, j, r1
    k += T
    while k <= total
        i = (k - 1) ÷ nb + 1
        j = (k - 1) % nb + 1
        ci, cj = coords[idx1[i]], coords[idx2[j]]
        rij = closest ? vector(ci, cj, boundary) : cj - ci
        d = norm(rij)
        better = is_min ? (d < best_d) : (d > best_d)
        if better
            best_d, best_i, best_j, best_disp = d, i, j, rij
        end
        k += T
    end
    out_dist[tid] = best_d
    out_i[tid] = best_i
    out_j[tid] = best_j
    out_disp[tid] = best_disp
end

@kernel inbounds=true function mindist_finalize_value_kernel!(dist_val, @Const(out_dist), ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    if tid == 1
        T = length(out_dist)
        best_d = out_dist[1]
        for k in 2:T
            d = out_dist[k]
            (is_min ? (d < best_d) : (d > best_d)) && (best_d = d)
        end
        dist_val[1] = best_d
    end
end

# Clears every candidate atom's `grad` entry in parallel. Must run (guaranteed by launch order on
# the same backend queue -- see mindist_gradient_fused! below) before mindist_finalize_grad_kernel!
# writes the new winning pair's 2 nonzero entries, or it would wipe them out again.
@kernel inbounds=true function mindist_clear_grad_kernel!(grad, @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    na = length(idx1)
    z = zero(eltype(grad))
    if tid <= na
        grad[idx1[tid]] = z
    else
        grad[idx2[tid - na]] = z
    end
end

@kernel inbounds=true function mindist_finalize_grad_kernel!(grad, d_buf, winner_i, winner_j, r_ij_buf,
                                                              @Const(out_dist), @Const(out_i), @Const(out_j),
                                                              @Const(out_disp), @Const(idx1), @Const(idx2),
                                                              ::Val{is_min}) where is_min
    tid = @index(Global, Linear)
    if tid == 1
        T = length(out_dist)
        best_d, best_slot = out_dist[1], 1
        for k in 2:T
            d = out_dist[k]
            (is_min ? (d < best_d) : (d > best_d)) && ((best_d, best_slot) = (d, k))
        end
        best_i, best_j, best_r = out_i[best_slot], out_j[best_slot], out_disp[best_slot]

        d_buf[1] = best_d
        winner_i[1] = best_i
        winner_j[1] = best_j
        r_ij_buf[1] = best_r
        if best_d > zero(best_d)
            dir = best_r / best_d
            grad[idx1[best_i]] = -dir
            grad[idx2[best_j]] = dir
        end
    end
end

function mindist_calculate_cv_fused!(dist_val, scratch::MinMaxScratch, coords, boundary, closest::Bool,
                                     is_min::Val)
    backend = get_backend(coords)
    T = length(scratch.out_dist)
    kernel_a! = mindist_tile_kernel!(backend, min(T, 256))
    kernel_a!(scratch.out_dist, scratch.out_i, scratch.out_j, scratch.out_disp, coords,
             scratch.idx1_dev, scratch.idx2_dev, boundary, closest, is_min; ndrange=T)
    kernel_b! = mindist_finalize_value_kernel!(backend, 1)
    kernel_b!(dist_val, scratch.out_dist, is_min; ndrange=1)
    return nothing
end

# Zero host syncs except one small readback when `extremal_cache` is supplied, to populate it for
# calculate_virial_dist!'s reuse. That readback is safe even though virial steps run outside CUDA
# graph capture entirely (simulators.jl), so it never needs to be graph-legal.
function mindist_gradient_fused!(grad, d_buf, scratch::MinMaxScratch, coords, boundary, closest::Bool,
                                 is_min::Val, extremal_cache)
    backend = get_backend(coords)
    T = length(scratch.out_dist)
    kernel_a! = mindist_tile_kernel!(backend, min(T, 256))
    kernel_a!(scratch.out_dist, scratch.out_i, scratch.out_j, scratch.out_disp, coords,
             scratch.idx1_dev, scratch.idx2_dev, boundary, closest, is_min; ndrange=T)

    na, nb = length(scratch.idx1_dev), length(scratch.idx2_dev)
    kernel_clear! = mindist_clear_grad_kernel!(backend, min(na + nb, 256))
    kernel_clear!(grad, scratch.idx1_dev, scratch.idx2_dev; ndrange=na + nb)

    kernel_b! = mindist_finalize_grad_kernel!(backend, 1)
    kernel_b!(grad, d_buf, scratch.winner_i, scratch.winner_j, scratch.r_ij, scratch.out_dist, scratch.out_i,
             scratch.out_j, scratch.out_disp, scratch.idx1_dev, scratch.idx2_dev, is_min; ndrange=1)

    if extremal_cache !== nothing
        extremal_cache.valid = true
        extremal_cache.i, extremal_cache.j = only(from_device(scratch.winner_i)), only(from_device(scratch.winner_j))
        extremal_cache.d, extremal_cache.r_ij = only(from_device(d_buf)), only(from_device(scratch.r_ij))
    end
    return nothing
end

function check_calc_type(calc_type)
    if !(calc_type in (:closest, :raw))
        throw(ArgumentError("calc_type argument must be :closest or :raw, found $calc_type"))
    end
end

"""
    CalcMinDist(calc_type=:closest)

Bias the minimum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMinDist
    calc_type::Symbol

    function CalcMinDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMinDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(md, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(md::CalcMinDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    _, _, d, _ = extremal_pair(coords_1, coords_2, md.calc_type, findmin, boundary)
    dist_val .= d
    return nothing
end

"""
    CalcMaxDist(calc_type=:closest)

Bias the maximum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMaxDist
    calc_type::Symbol

    function CalcMaxDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMaxDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(md, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(md::CalcMaxDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    _, _, d, _ = extremal_pair(coords_1, coords_2, md.calc_type, findmax, boundary)
    dist_val .= d
    return nothing
end

"""
    CalcCMDist(calc_type=:closest)

Bias the distance between the centers of mass of two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

Should generally be used with molecule unwrapping since it assumes that the atoms
within each group are in the same periodic box.
If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcCMDist
    calc_type::Symbol

    function CalcCMDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(cd::CalcCMDist, coords_1, coords_2, boundary,
                             atoms_1, atoms_2, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(cd, coords_1, coords_2, dist_val, boundary, atoms_1, atoms_2, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(cd::CalcCMDist, coords_1, coords_2, dist_val, boundary,
                              atoms_1, atoms_2, args...; kwargs...)
    com_1 = similar(coords_1, 1)
    com_2 = similar(coords_2, 1)
    center_of_mass!(coords_1, atoms_1, com_1)
    center_of_mass!(coords_2, atoms_2, com_2)
    if cd.calc_type == :closest
        dist_val .= norm.(vector.(com_1, com_2, (boundary,)))
    else
        dist_val .= norm.(com_2 .- com_1)
    end
    return nothing
end

# Fused CalcCMDist path: reduce (parallel partial mass/weighted-position sums), finalize (sum
# the small partials), grad-write (per-atom, parallel). A serial ndrange=1 thread would leave
# the device idle and scaled linearly with group size. calculate_cv! needs reduce+finalize;
# cv_gradient! needs all three.
mutable struct CMDistScratch{IV, MV, WV, DV, SV}
    idx1_dev::IV
    idx2_dev::IV
    partial_mass1::MV
    partial_wpos1::WV
    partial_mass2::MV
    partial_wpos2::WV
    dir_buf::DV
    mtot1_buf::SV
    mtot2_buf::SV
end


@kernel inbounds=true function cmdist_reduce_kernel!(pmass1, pwpos1, pmass2, pwpos2,
                                                      @Const(coords), @Const(atoms),
                                                      @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    T1 = length(pmass1)
    if tid <= T1
        na = length(idx1)
        acc, mtot = zero(eltype(pwpos1)), zero(eltype(pmass1))
        k = tid
        while k <= na
            mk = mass(atoms[idx1[k]])
            acc += coords[idx1[k]] * mk
            mtot += mk
            k += T1
        end
        pmass1[tid] = mtot
        pwpos1[tid] = acc
    end
    T2 = length(pmass2)
    if tid <= T2
        nb = length(idx2)
        acc, mtot = zero(eltype(pwpos2)), zero(eltype(pmass2))
        k = tid
        while k <= nb
            mk = mass(atoms[idx2[k]])
            acc += coords[idx2[k]] * mk
            mtot += mk
            k += T2
        end
        pmass2[tid] = mtot
        pwpos2[tid] = acc
    end
end

@kernel inbounds=true function cmdist_finalize_kernel!(dist_val, dir_buf, mtot1_buf, mtot2_buf,
                                                        @Const(pmass1), @Const(pwpos1),
                                                        @Const(pmass2), @Const(pwpos2),
                                                        boundary, closest::Bool)
    tid = @index(Global, Linear)
    if tid == 1
        T1, T2 = length(pmass1), length(pmass2)
        mtot1, wpos1 = pmass1[1], pwpos1[1]
        for k in 2:T1
            mtot1 += pmass1[k]
            wpos1 += pwpos1[k]
        end
        mtot2, wpos2 = pmass2[1], pwpos2[1]
        for k in 2:T2
            mtot2 += pmass2[k]
            wpos2 += pwpos2[k]
        end
        com1, com2 = wpos1 / mtot1, wpos2 / mtot2
        r12 = closest ? vector(com1, com2, boundary) : com2 - com1
        d = norm(r12)
        dist_val[1] = d
        # Unconditional division: cmdist_grad_write_kernel! only reads dir_buf inside its own
        # `d > 0` branch, so a NaN/Inf here when d==0 is never observed.
        dir_buf[1] = r12 / d
        mtot1_buf[1] = mtot1
        mtot2_buf[1] = mtot2
    end
end

@kernel inbounds=true function cmdist_grad_write_kernel!(grad, @Const(d_buf), @Const(dir_buf),
                                                          @Const(mtot1_buf), @Const(mtot2_buf),
                                                          @Const(atoms), @Const(idx1), @Const(idx2))
    tid = @index(Global, Linear)
    na = length(idx1)
    d = d_buf[1]
    if d > zero(d)
        dir = dir_buf[1]
        if tid <= na
            grad[idx1[tid]] = -dir * (mass(atoms[idx1[tid]]) / mtot1_buf[1])
        else
            k = tid - na
            grad[idx2[k]] = dir * (mass(atoms[idx2[k]]) / mtot2_buf[1])
        end
    else
        z = zero(eltype(grad))
        if tid <= na
            grad[idx1[tid]] = z
        else
            k = tid - na
            grad[idx2[k]] = z
        end
    end
end

"""
    CalcSingleDist(calc_type=:closest)

Bias the distance between two atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcSingleDist
    calc_type::Symbol

    function CalcSingleDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(sd::CalcSingleDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_val = similar(coords_1, eltype(eltype(coords_1)), 1)
    dist_between_groups!(sd, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    return only(from_device(dist_val))
end

function dist_between_groups!(sd::CalcSingleDist, coords_1, coords_2, dist_val, boundary, args...; kwargs...)
    if length(coords_1) > 1 || length(coords_2) > 1
        throw(ArgumentError("CalcSingleDist can only be used with atom groups containing one atom"))
    end
    c1, c2 = only(from_device(coords_1)), only(from_device(coords_2))
    if sd.calc_type == :closest
        dist_val .= norm(vector(c1, c2, boundary))
    else
        dist_val .= norm(c2 - c1)
    end
end

"""
    CalcDist(atom_inds_1, atom_inds_2, dist_type=CalcMinDist(), correction=:pbc)

Bias the distance between two atoms or groups of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds_1`: indices of the atom(s) in the first group.
- `atom_inds_2`: indices of the atom(s) in the second group.
- `dist_type=CalcMinDist()`: type of distance to calculate.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. If using multiple atoms in
    a group, they should generally be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcDist{DT}
    atom_inds_1::Vector{Int}
    atom_inds_2::Vector{Int}
    dist_type::DT
    correction::Symbol
    has_virial::Bool

    function CalcDist(atom_inds_1, atom_inds_2, dist_type::DT=CalcMinDist(),
                      correction=:pbc, has_virial = true) where DT
        check_correction_arg(correction)
        return new{DT}(atom_inds_1, atom_inds_2, dist_type, correction, has_virial)
    end
end

"""
    calculate_cv(cv, coords, atoms, boundary, velocities; kwargs...)

Calculate the value of a collective variable (CV) with the current system state.

New CV types should implement this function.
This function does not apply the molecule correction over the boundaries; if
required, `coords` can be obtained from `unwrap_molecules` first.
The gradient of this function with respect to coordinates, used to calculate forces,
is by default calculated with automatic differentiation when Enzyme is imported.
Alternatively, the `cv_gradient` function can be defined for a new CV type.
"""
function calculate_cv(cv::CalcDist, coords, atoms, boundary, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
    return only(from_device(buff))
end

"""
    calculate_cv!(cv, coords, atoms, boundary, buff, velocities; kwargs...)

Mutating counterpart to [`calculate_cv`](@ref): writes the CV value into the preallocated
1-element `buff` instead of allocating and returning it.
"""
function calculate_cv!(cv::CalcDist, coords, atoms, boundary, buff, args...; kwargs...)
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# Fused path used whenever a persistent `MinMaxScratch` is supplied and `coords` is GPU-resident
# (see MinMaxScratch's docstring); falls back to the generic method otherwise (CPU, or no scratch).
function calculate_cv!(cv::CalcDist{<:Union{CalcMinDist, CalcMaxDist}}, coords, atoms, boundary, buff,
                       args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        is_min = cv.dist_type isa CalcMinDist
        mindist_calculate_cv_fused!(buff, scratch, coords, boundary, closest, Val(is_min))
        return nothing
    end
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# Fused path used whenever a persistent `CMDistScratch` is supplied and `coords` is GPU-resident;
# falls back to the generic method otherwise (CPU, or no scratch).
function calculate_cv!(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, buff, args...;
                       scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        backend = get_backend(coords)
        T1, T2 = length(scratch.partial_mass1), length(scratch.partial_mass2)
        reduce! = cmdist_reduce_kernel!(backend, min(max(T1, T2), 256))
        reduce!(scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
               coords, atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=max(T1, T2))
        finalize! = cmdist_finalize_kernel!(backend, 1)
        finalize!(buff, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
                  scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
                  boundary, closest; ndrange=1)
        return nothing
    end
    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]
    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]
    dist_between_groups!(cv.dist_type, coords_1, coords_2, buff, boundary, atoms_1, atoms_2; kwargs...)
    return nothing
end

# Single-thread kernel avoiding the generic broadcast path's multiple kernel-launch overhead.
@kernel inbounds=true function single_dist_cv_kernel!(d_buf, @Const(coords), i, j, boundary,
                                                       closest::Bool)
    idx = @index(Global, Linear)
    if idx == 1
        r_ij = closest ? vector(coords[i], coords[j], boundary) : coords[j] - coords[i]
        d_buf[1] = norm(r_ij)
    end
end

@kernel inbounds=true function single_dist_cv_gradient_kernel!(grad, d_buf, @Const(coords),
                                                                i, j, boundary, closest::Bool)
    idx = @index(Global, Linear)
    if idx == 1
        r_ij = closest ? vector(coords[i], coords[j], boundary) : coords[j] - coords[i]
        d = norm(r_ij)
        d_buf[1] = d
        if d > zero(d)
            dir = r_ij / d
            grad[i] = -dir
            grad[j] = dir
        else
            z = zero(r_ij) / oneunit(d)
            grad[i] = z
            grad[j] = z
        end
    end
end

function calculate_cv!(cv::CalcDist{CalcSingleDist}, coords::AbstractGPUArray, atoms, boundary,
                       buff::AbstractGPUArray, args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    closest = cv.dist_type.calc_type == :closest
    backend = get_backend(coords)
    kernel! = single_dist_cv_kernel!(backend, 1)
    kernel!(buff, coords, i, j, boundary, closest; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the distance between two atoms.
#
# Mathematics:
# Let the coordinates of the two atoms be r_i and r_j.
# The minimum image vector from atom i to atom j is r_{ij} = r_j - r_i.
# The distance is given by d = |r_{ij}|.
# The gradients with respect to the atomic coordinates are:
# ∇_{r_i} d = -r_{ij}/d,    ∇_{r_j} d = r_{ij}/d

@doc raw"""
    cv_gradient(cv, coords, atoms, boundary, velocities; kwargs...)

Calculates the analytical gradient of a collective variable (CV) with respect to the
system coordinates.

Returns a tuple containing the gradient (as an array of vectors) and the current
value of the CV.
When Enzyme is imported this defaults to using AD, but an explicit method
can be provided for a given CV type and is defined for built-in CVs.
The AD approach should work with and without units.

Supported CV Types:
- `CalcDist`: Distance between two atoms or groups (Min, Max, Center of Mass, or Single).
- `CalcRg`: Radius of gyration of a group of atoms.
- `CalcRMSD`: Root-mean-square deviation from a reference structure using Kabsch alignment.
- `CalcTorsion`: Torsion (dihedral) angle defined by four atoms.

Allocates the gradient array and a 1-element CV-value buffer, then delegates to
[`cv_gradient!`](@ref), which writes into them in place. Call `cv_gradient!` directly
with reused buffers to avoid the per-call allocation (e.g. across repeated timesteps).
"""
function cv_gradient(cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

"""
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, velocities; kwargs...)

Mutating counterpart to [`cv_gradient`](@ref): writes the gradient into the preallocated
`grad` (same shape/backend as `coords`) and the CV value into the preallocated 1-element
`d_buf`, instead of allocating and returning them.
"""
function cv_gradient!(grad, d_buf, cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    c1 = @view coords[i:i]
    c2 = @view coords[j:j]

    r_ij = cv.dist_type.calc_type == :closest ? vector.(c1, c2, (boundary,)) : c2 .- c1
    d = norm.(r_ij)
    d_buf .= d

    mask = d .> zero(eltype(d))
    d_safe = ifelse.(mask, d, oneunit.(d))
    dir = r_ij ./ d_safe
    grad[i:i] .= ifelse.(mask, .-dir, zero(dir))
    grad[j:j] .= ifelse.(mask, dir, zero(dir))

    return nothing
end

# GPU fast path: one kernel launch
function cv_gradient!(grad::AbstractGPUArray, d_buf::AbstractGPUArray,
                      cv::CalcDist{CalcSingleDist}, coords::AbstractGPUArray, atoms, boundary,
                      args...; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    closest = cv.dist_type.calc_type == :closest
    backend = get_backend(coords)
    kernel! = single_dist_cv_gradient_kernel!(backend, 1)
    kernel!(grad, d_buf, coords, i, j, boundary, closest; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the minimum distance between two groups of atoms.
#
# Mathematics:
# Let A and B be two sets of atoms.
# The minimum distance is defined by the specific pair (i*, j*) ∈ A x B that 
# minimizes d_{i,j} = |r_{i,j}|.
# The gradient evaluates to zero for all atoms except i* and j*, for which it reduces 
# to the single distance gradient:
# ∇_{r_{i*}} d = -r_{i*j*}/d,    ∇_{r_{j*}} d = r_{i*j*}/d
function cv_gradient(cv::CalcDist{CalcMinDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcMinDist}, coords, atoms, boundary, args...;
                      extremal_cache=nothing, scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        mindist_gradient_fused!(grad, d_buf, scratch, coords, boundary, cv.dist_type.calc_type == :closest,
                                Val(true), extremal_cache)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    i, j, d, r_ij = extremal_pair(c1, c2, cv.dist_type.calc_type, findmin, boundary)
    d_buf .= d
    if extremal_cache !== nothing
        extremal_cache.valid, extremal_cache.i, extremal_cache.j = true, i, j
        extremal_cache.d, extremal_cache.r_ij = d, r_ij
    end

    # necessary to clear the whole candidate set to remove previous results
    zg = zero(eltype(grad))
    grad[cv.atom_inds_1] .= (zg,)
    grad[cv.atom_inds_2] .= (zg,)

    if d > zero(d)
        dir = r_ij / d
        gi, gj = cv.atom_inds_1[i], cv.atom_inds_2[j]
        grad[gi:gi] .= (-dir,)
        grad[gj:gj] .= (dir,)
    end

    return nothing
end

# Computes the analytical gradient of the maximum distance between two groups of atoms.
#
# Mathematics:
# Let A and B be two sets of atoms.
# The maximum distance is defined by the specific pair (i*, j*) ∈ A x B that 
# maximizes d_{i,j} = |r_{i,j}|.
# The gradient is equivalent to the single distance gradient applied exclusively 
# to this maximizing pair.
function cv_gradient(cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, args...;
                      extremal_cache=nothing, scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        mindist_gradient_fused!(grad, d_buf, scratch, coords, boundary, cv.dist_type.calc_type == :closest,
                                Val(false), extremal_cache)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    i, j, d, r_ij = extremal_pair(c1, c2, cv.dist_type.calc_type, findmax, boundary)
    d_buf .= d
    if extremal_cache !== nothing
        extremal_cache.valid, extremal_cache.i, extremal_cache.j = true, i, j
        extremal_cache.d, extremal_cache.r_ij = d, r_ij
    end

    # See CalcMinDist's cv_gradient! for why this clear is needed with a reused `grad` buffer.
    zg = zero(eltype(grad))
    grad[cv.atom_inds_1] .= (zg,)
    grad[cv.atom_inds_2] .= (zg,)

    if d > zero(d)
        dir = r_ij / d
        gi, gj = cv.atom_inds_1[i], cv.atom_inds_2[j]
        grad[gi:gi] .= (-dir,)
        grad[gj:gj] .= (dir,)
    end

    return nothing
end

# Computes the analytical gradient of the center-of-mass distance between two groups of atoms.
#
# Mathematics:
# Let M_A and M_B be the total masses of groups A and B.
# Let R_A and R_B be their respective centers of mass, and D = |R_B - R_A|.
# Applying the chain rule through the center of mass definition, the gradients 
# for individual atoms are proportional to their fractional mass:
# ∇_{r_i} D = -(m_i/M_A) * (R_{AB}/D)    ∀ i ∈ A
# ∇_{r_j} D =  (m_j/M_B) * (R_{AB}/D)    ∀ j ∈ B
function cv_gradient(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcDist{CalcCMDist}, coords, atoms, boundary, args...;
                      scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        closest = cv.dist_type.calc_type == :closest
        backend = get_backend(coords)
        T1, T2 = length(scratch.partial_mass1), length(scratch.partial_mass2)
        reduce! = cmdist_reduce_kernel!(backend, min(max(T1, T2), 256))
        reduce!(scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
               coords, atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=max(T1, T2))
        finalize! = cmdist_finalize_kernel!(backend, 1)
        finalize!(d_buf, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
                  scratch.partial_mass1, scratch.partial_wpos1, scratch.partial_mass2, scratch.partial_wpos2,
                  boundary, closest; ndrange=1)
        na, nb = length(scratch.idx1_dev), length(scratch.idx2_dev)
        write! = cmdist_grad_write_kernel!(backend, min(na + nb, 256))
        write!(grad, d_buf, scratch.dir_buf, scratch.mtot1_buf, scratch.mtot2_buf,
              atoms, scratch.idx1_dev, scratch.idx2_dev; ndrange=na + nb)
        return nothing
    end

    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1_buf = similar(c1, 1)
    com2_buf = similar(c2, 1)
    center_of_mass!(c1, a1, com1_buf)
    center_of_mass!(c2, a2, com2_buf)

    if cv.dist_type.calc_type == :closest
        r_12 = vector.(com1_buf, com2_buf, (boundary,))
    else
        r_12 = com2_buf .- com1_buf
    end

    d = norm.(r_12)
    d_buf .= d

    mask = d .> zero(eltype(d))
    d_safe = ifelse.(mask, d, oneunit.(d))
    dir = r_12 ./ d_safe

    m1, m2 = mass.(a1), mass.(a2)
    # sum(...; dims=1), not sum(...): stays device-resident (a 1-element array) instead of
    # forcing a blocking device->host sync -- see the same note on center_of_mass! above.
    M1, M2 = sum(m1; dims=1), sum(m2; dims=1)

    grad1 = (.-dir) .* (m1 ./ M1)
    grad2 = dir .* (m2 ./ M2)
    # @inbounds: cv.atom_inds_1/atom_inds_2 are validated at CV-construction time, always valid
    # indices into `grad` -- without it, GPUArrays' fancy-index setindex! bounds check
    # (checkindex -> all(...)) is an *extra* host sync on top of the actual write, and (found
    # directly, verifying CUDA graph capture) raises a device-side exception if this runs inside
    # a captured region at all.
    @inbounds grad[cv.atom_inds_1] = ifelse.(mask, grad1, zero.(grad1))
    @inbounds grad[cv.atom_inds_2] = ifelse.(mask, grad2, zero.(grad2))

    return nothing
end

function calculate_virial!(virial_buff, cv::CalcDist, coords, forces, atoms, boundary;
                           precomputed_extremum=nothing, kwargs...)
    calculate_virial_dist!(virial_buff, cv.dist_type, cv, coords, forces, atoms, boundary;
                           precomputed_extremum=precomputed_extremum)
end

function calculate_virial_dist!(virial_buff, dt::CalcSingleDist, cv, coords, forces, atoms, boundary;
                                kwargs...)
    i = cv.atom_inds_1[1]
    j = cv.atom_inds_2[1]
    f_i = only(from_device(forces[i:i]))
    c_i = only(from_device(coords[i:i]))
    c_j = only(from_device(coords[j:j]))

    if dt.calc_type == :closest
        r_ji = vector(c_j, c_i, boundary)
    else
        r_ji = c_i - c_j
    end

    virial_buff .+= r_ji * transpose(f_i)
end

# `precomputed_extremum`, if a valid `ExtremalPairCache` from this timestep's `cv_gradient!` call,
# skips recomputing the O(group_a * group_b) extremal search.
function calculate_virial_dist!(virial_buff, dt::CalcMinDist, cv, coords, forces, atoms, boundary;
                                precomputed_extremum=nothing, kwargs...)
    if precomputed_extremum !== nothing && precomputed_extremum.valid
        r_ij = precomputed_extremum.r_ij
    else
        c1 = @view coords[cv.atom_inds_1]
        c2 = @view coords[cv.atom_inds_2]
        _, _, _, r_ij = extremal_pair(c1, c2, dt.calc_type, findmin, boundary)
    end
    r_ji = -r_ij

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcMaxDist, cv, coords, forces, atoms, boundary;
                                precomputed_extremum=nothing, kwargs...)
    if precomputed_extremum !== nothing && precomputed_extremum.valid
        r_ij = precomputed_extremum.r_ij
    else
        c1 = @view coords[cv.atom_inds_1]
        c2 = @view coords[cv.atom_inds_2]
        _, _, _, r_ij = extremal_pair(c1, c2, dt.calc_type, findmax, boundary)
    end
    r_ji = -r_ij

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcCMDist, cv, coords, forces, atoms, boundary;
                                kwargs...)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1_buf = similar(c1, 1)
    com2_buf = similar(c2, 1)
    center_of_mass!(c1, a1, com1_buf)
    center_of_mass!(c2, a2, com2_buf)
    com1, com2 = only(from_device(com1_buf)), only(from_device(com2_buf))

    if dt.calc_type == :closest
        r_12 = vector(com2, com1, boundary)
    else
        r_12 = com1 - com2
    end

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_12 * transpose(f_sum)
end

"""
    CalcRg(atom_inds=[], correction=:pbc)

Bias the radius of gyration of a group of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcRg
    atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool

    function CalcRg(atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        return new(atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, atoms, buff, args...; kwargs...)
    return only(from_device(buff))
end

# Fused path for CalcRg's calculate_cv!/cv_gradient!, used whenever a persistent `RgScratch` is
# supplied and `coords` is GPU-resident. Rg has a genuine 2-stage dependency: the sum of squared
# deviations from the center of mass needs the already-finalized COM as an input. Hence 2
# reduce+finalize pairs back to back -- COM reduce/finalize, then an Isum reduce/finalize using
# the now-known COM -- plus a final grad-write kernel (gradient path only) that writes each
# atom's entry in parallel. `calculate_cv!` needs the first 4 stages; `cv_gradient!` needs all 5.
const RG_TILE_CAP = 1024

mutable struct RgScratch{IV, MV, WV, IsV, CV, MtV}
    idx_dev::IV
    partial_mass::MV
    partial_wpos::WV
    partial_isum::IsV
    com_buf::CV
    mtot_buf::MtV
end

@kernel inbounds=true function rg_com_reduce_kernel!(pmass, pwpos, @Const(coords), @Const(atoms), @Const(idx))
    tid = @index(Global, Linear)
    T = length(pmass)
    n = length(idx)
    acc, mtot = zero(eltype(pwpos)), zero(eltype(pmass))
    k = tid
    while k <= n
        mk = mass(atoms[idx[k]])
        acc += coords[idx[k]] * mk
        mtot += mk
        k += T
    end
    pmass[tid] = mtot
    pwpos[tid] = acc
end

@kernel inbounds=true function rg_com_finalize_kernel!(com_buf, mtot_buf, @Const(pmass), @Const(pwpos))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pmass)
        mtot, wpos = pmass[1], pwpos[1]
        for k in 2:T
            mtot += pmass[k]
            wpos += pwpos[k]
        end
        com_buf[1] = wpos / mtot
        mtot_buf[1] = mtot
    end
end

# Two separate kernels, not one with a boundary/use_pbc flag: calculate_cv! has no `boundary` to
# pass (matches radius_gyration's CPU definition, no PBC correction), while cv_gradient! does.
# This value/gradient asymmetry predates this rework and is preserved as-is.
@kernel inbounds=true function rg_isum_reduce_value_kernel!(pisum, @Const(coords), @Const(atoms),
                                                             @Const(idx), @Const(com_buf))
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(idx)
    com = com_buf[1]
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(coords[idx[k]] - com) * mass(atoms[idx[k]])
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rg_isum_reduce_grad_kernel!(pisum, @Const(coords), @Const(atoms),
                                                            @Const(idx), @Const(com_buf), boundary)
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(idx)
    com = com_buf[1]
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(vector(com, coords[idx[k]], boundary)) * mass(atoms[idx[k]])
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rg_finalize_value_kernel!(dist_val, @Const(pisum), @Const(mtot_buf))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        dist_val[1] = sqrt(Isum / mtot_buf[1])
    end
end

@kernel inbounds=true function rg_finalize_grad_kernel!(d_buf, @Const(pisum), @Const(mtot_buf))
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        d_buf[1] = sqrt(Isum / mtot_buf[1])
    end
end

@kernel inbounds=true function rg_grad_write_kernel!(grad, @Const(d_buf), @Const(coords), @Const(atoms),
                                                      @Const(idx), @Const(com_buf), @Const(mtot_buf), boundary)
    tid = @index(Global, Linear)
    rg = d_buf[1]
    if rg > zero(rg)
        factor = 1 / (mtot_buf[1] * rg)
        grad[idx[tid]] = factor * mass(atoms[idx[tid]]) * vector(com_buf[1], coords[idx[tid]], boundary)
    else
        grad[idx[tid]] = zero(eltype(grad))
    end
end

function calculate_cv!(cv::CalcRg, coords, atoms, buff, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        T = length(scratch.partial_mass)
        reduce_com! = rg_com_reduce_kernel!(backend, min(T, 256))
        reduce_com!(scratch.partial_mass, scratch.partial_wpos, coords, atoms, scratch.idx_dev; ndrange=T)
        finalize_com! = rg_com_finalize_kernel!(backend, 1)
        finalize_com!(scratch.com_buf, scratch.mtot_buf, scratch.partial_mass, scratch.partial_wpos; ndrange=1)
        reduce_isum! = rg_isum_reduce_value_kernel!(backend, min(T, 256))
        reduce_isum!(scratch.partial_isum, coords, atoms, scratch.idx_dev, scratch.com_buf; ndrange=T)
        finalize_val! = rg_finalize_value_kernel!(backend, 1)
        finalize_val!(buff, scratch.partial_isum, scratch.mtot_buf; ndrange=1)
        return nothing
    end
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    buff .= radius_gyration(coords_used, atoms_used)
    return nothing
end

# Computes the analytical gradient of the radius of gyration.
#
# Mathematics:
# The mass-weighted radius of gyration is:
# R_g = sqrt( (1/M) * Σ_k m_k |r_k - R_COM|^2 )
# 
# Differentiating with respect to the coordinates of atom k yields:
# ∇_{r_k} R_g = [m_k / (M * R_g)] * (r_k - R_COM)
#
# Note: The derivative of the center of mass R_COM with respect to r_k cancels out
# in the summation due to the definition of the center of mass.
function cv_gradient(cv::CalcRg, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcRg, coords, atoms, boundary, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        T = length(scratch.partial_mass)
        reduce_com! = rg_com_reduce_kernel!(backend, min(T, 256))
        reduce_com!(scratch.partial_mass, scratch.partial_wpos, coords, atoms, scratch.idx_dev; ndrange=T)
        finalize_com! = rg_com_finalize_kernel!(backend, 1)
        finalize_com!(scratch.com_buf, scratch.mtot_buf, scratch.partial_mass, scratch.partial_wpos; ndrange=1)
        reduce_isum! = rg_isum_reduce_grad_kernel!(backend, min(T, 256))
        reduce_isum!(scratch.partial_isum, coords, atoms, scratch.idx_dev, scratch.com_buf, boundary; ndrange=T)
        finalize_grad! = rg_finalize_grad_kernel!(backend, 1)
        finalize_grad!(d_buf, scratch.partial_isum, scratch.mtot_buf; ndrange=1)
        n = length(scratch.idx_dev)
        write! = rg_grad_write_kernel!(backend, min(n, 256))
        write!(grad, d_buf, coords, atoms, scratch.idx_dev, scratch.com_buf, scratch.mtot_buf, boundary; ndrange=n)
        return nothing
    end

    atom_inds_used = iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds
    c_used = @view coords[atom_inds_used]
    a_used = @view atoms[atom_inds_used]

    com_buf = similar(c_used, 1)
    center_of_mass!(c_used, a_used, com_buf)
    m_used = mass.(a_used)
    # sum(...; dims=1), not sum(...): stays device-resident -- see the center_of_mass! note above.
    M_total = sum(m_used; dims=1)

    r_ic_all = vector.(com_buf, c_used, (boundary,))
    rg_sq = sum(sum_abs2.(r_ic_all) .* m_used; dims=1) ./ M_total
    rg = sqrt.(rg_sq)
    d_buf .= rg

    # use mask to avoid host-sync
    mask = rg .> zero(eltype(rg))
    rg_safe = ifelse.(mask, rg, oneunit.(rg))
    inv_factor = 1 ./ (M_total .* rg_safe)
    factor = ifelse.(mask, inv_factor, zero.(inv_factor))
    grad[atom_inds_used] .= factor .* m_used .* r_ic_all

    return nothing
end

# For Rg and also for the RMSD the forces applied to the atoms 
# are dependent only on the relative configuration of said
# atoms, making them translationally invariant. Therefore:
#
# Σ F_i = 0
#
# We can exploit this fact to obtain the virial by computing 
#
# Ξ = Σ (r_i - r_COM) ⊗ F_i; 
#
# rearranging:
#
# Ξ = Σ ( r_i ⊗ F_i ) - r_COM ⊗ Σ F_i = Σ r_i ⊗ F_i
#
# which is equivalent to the standard definition of the virial!
# Note: we cannot just compute Σ r_i ⊗ F_i as this will give 
# different results depending on the choice of origin of coordinates.

function calculate_virial!(virial_buff, cv::CalcRg, coords, forces, atoms, boundary; kwargs...)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    a_used = @view atoms[ids]

    # Calculate Center of Mass of the group to define relative coordinates
    com_buf = similar(c_used, 1)
    center_of_mass!(c_used, a_used, com_buf)

    # Accumulate sum( (r_i - r_com) * F_i^T )
    r_ic_all = vector.(com_buf, c_used, (boundary,))
    virial_buff .+= sum(r_ic_all .* transpose.(f_used))
end

"""
    CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc)

Bias the root-mean-square deviation (RMSD) between the coordinates of a group of atoms
and a set of reference coordinates.

Given as an argument to [`BiasPotential`](@ref).
The two sets of coordinates are superimposed using the Kabsch algorithm.

# Arguments
- `ref_coords`: reference coordinates. Should be constructed with an array type matching the
    `System` this CV will be used with (a plain `Array` for CPU, or the same GPU array type,
    e.g. `CuArray`/`ROCArray`, as the system's coordinates for GPU) — this is not converted
    automatically, the same convention already implicitly expected of `atoms`/`coords`/
    `velocities` elsewhere.
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `ref_atom_inds=[]`: indices of the reference coordinates to use, `[]` uses all coordinates.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native
    bonded-topology traversal.
"""
struct CalcRMSD{RC}
    ref_coords::RC
    atom_inds::Vector{Int}
    ref_atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool

    function CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        RC = typeof(ref_coords)
        new{RC}(ref_coords, atom_inds, ref_atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcRMSD, coords, args...; kwargs...)
    buff = similar(coords, eltype(eltype(coords)), 1)
    calculate_cv!(cv, coords, buff, args...; kwargs...)
    return only(from_device(buff))
end

const RMSD_TILE_CAP = 1024

"""
    RmsdScratch

Persistent GPU scratch for the fused CalcRMSD `calculate_cv!`/`cv_gradient!` path.
`idx_dev`/`coords_used` avoid re-gathering/re-uploading the used atom indices every call;
`ref_coords_used`/`ref_kabsch` precompute the reference side once (it never changes after
construction); a reduce/finalize/grad-write kernel trio computes the RMSD value and gradient
device-side once the Kabsch rotation is known.
"""
mutable struct RmsdScratch{IV, CV, RCV, KV, PV}
    idx_dev::IV
    coords_used::CV
    ref_coords_used::RCV
    ref_kabsch::KV
    partial_isum::PV
end

@kernel inbounds=true function gather_kernel!(dst, @Const(src), @Const(idx))
    k = @index(Global, Linear)
    if k <= length(idx)
        dst[k] = src[idx[k]]
    end
end

@kernel inbounds=true function rmsd_isum_reduce_kernel!(pisum, @Const(ref_used), @Const(coords_used),
                                                         rot, trans_1, trans_2)
    tid = @index(Global, Linear)
    T = length(pisum)
    n = length(ref_used)
    acc = zero(eltype(pisum))
    k = tid
    while k <= n
        acc += sum_abs2(rot * (ref_used[k] - trans_1) - (coords_used[k] - trans_2))
        k += T
    end
    pisum[tid] = acc
end

@kernel inbounds=true function rmsd_finalize_value_kernel!(dist_val, @Const(pisum), n)
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        dist_val[1] = sqrt(Isum / n)
    end
end

@kernel inbounds=true function rmsd_finalize_grad_kernel!(d_buf, @Const(pisum), n)
    tid = @index(Global, Linear)
    if tid == 1
        T = length(pisum)
        Isum = pisum[1]
        for k in 2:T
            Isum += pisum[k]
        end
        d_buf[1] = sqrt(Isum / n)
    end
end

@kernel inbounds=true function rmsd_grad_write_kernel!(grad, @Const(d_buf), @Const(ref_used), @Const(coords_used),
                                                        @Const(idx), rot, trans_1, trans_2)
    tid = @index(Global, Linear)
    rmsd_val = d_buf[1]
    if rmsd_val > zero(rmsd_val)
        n = length(idx)
        factor = 1 / (n * rmsd_val)
        diff_k = rot * (ref_used[tid] - trans_1) - (coords_used[tid] - trans_2)
        grad[idx[tid]] = -factor * diff_k
    else
        grad[idx[tid]] = zero(eltype(grad))
    end
end

function calculate_cv!(cv::CalcRMSD, coords, buff, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        n = length(scratch.idx_dev)
        kernel! = gather_kernel!(backend, min(n, 256))
        kernel!(scratch.coords_used, coords, scratch.idx_dev; ndrange=n)
        rot, trans_1, trans_2 = kabsch_rotation_nograd(scratch.ref_coords_used, scratch.coords_used;
                                                        cached_1=scratch.ref_kabsch)
        T = length(scratch.partial_isum)
        reduce! = rmsd_isum_reduce_kernel!(backend, min(T, 256))
        reduce!(scratch.partial_isum, scratch.ref_coords_used, scratch.coords_used, rot, trans_1, trans_2; ndrange=T)
        finalize! = rmsd_finalize_value_kernel!(backend, 1)
        finalize!(buff, scratch.partial_isum, n; ndrange=1)
        return nothing
    end
    coords_used, ref_coords_used = rmsd_coords(cv, coords)
    buff .= rmsd(ref_coords_used, coords_used)
    return nothing
end

# Select the atoms of the system and of the reference used by a CalcRMSD collective variable
function rmsd_coords(cv::CalcRMSD, coords)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    ref_atom_inds_used = (iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords)
                                                           : cv.ref_atom_inds)
    return coords[atom_inds_used], cv.ref_coords[ref_atom_inds_used]
end

function calculate_cv_ustrip!(unit_arr, args...)
    cv = calculate_cv(args...)
    # Enzyme requires a unitless value to be returned
    # We strip the unit, store it and add it back on later
    unit_arr[1] = unit(cv)
    return ustrip(cv)
end

# Computes the analytical gradient of the optimal Root-Mean-Square Deviation (RMSD)
# using Kabsch alignment.
#
# Mathematics:
# Let r_k^{sys} be the current system coordinates and R_COM^{sys} be their centroid. 
# Let r_k^{ref} be the centered reference coordinates.
# The optimally aligned RMSD distance is:
# d_{RMSD} = sqrt( (1/N) * Σ_{k=1}^N |(r_k^{sys} - R_COM^{sys}) - Q r_k^{ref}|^2 )
#
# Because the rotation matrix Q optimally minimizes the distance, the derivative of Q 
# with respect to coordinates vanishes.
# The exact analytical gradient for an evaluated atom k simplifies to:
# ∇_{r_k} d_{RMSD} = [1 / (N * d_{RMSD})] * ((r_k^{sys} - R_COM^{sys}) - Q r_k^{ref})
function cv_gradient(cv::CalcRMSD, coords, args...; kwargs...)
    grad = ustrip_vec.(zero(coords))
    d_buf = similar(coords, eltype(eltype(coords)), 1)
    cv_gradient!(grad, d_buf, cv, coords, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcRMSD, coords, args...; scratch=nothing, kwargs...)
    if scratch !== nothing && is_gpu_resident(coords)
        backend = get_backend(coords)
        n = length(scratch.idx_dev)
        kernel! = gather_kernel!(backend, min(n, 256))
        kernel!(scratch.coords_used, coords, scratch.idx_dev; ndrange=n)
        rot, trans_1, trans_2 = kabsch_rotation_nograd(scratch.ref_coords_used, scratch.coords_used;
                                                        cached_1=scratch.ref_kabsch)
        T = length(scratch.partial_isum)
        reduce! = rmsd_isum_reduce_kernel!(backend, min(T, 256))
        reduce!(scratch.partial_isum, scratch.ref_coords_used, scratch.coords_used, rot, trans_1, trans_2; ndrange=T)
        finalize! = rmsd_finalize_grad_kernel!(backend, 1)
        finalize!(d_buf, scratch.partial_isum, n; ndrange=1)
        write! = rmsd_grad_write_kernel!(backend, min(n, 256))
        write!(grad, d_buf, scratch.ref_coords_used, scratch.coords_used, scratch.idx_dev,
              rot, trans_1, trans_2; ndrange=n)
        return nothing
    end

    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used, ref_c_used = rmsd_coords(cv, coords)
    N = length(c_used)

    # Deviations of the rotated reference from the current coordinates
    diffs = kabsch_deviations(ref_c_used, c_used)
    rmsd_val = sqrt(mean(sum_abs2, diffs))
    d_buf .= rmsd_val

    if rmsd_val > zero(rmsd_val)
        factor = 1 / (N * rmsd_val)
        grad[atom_inds_used] = (-factor,) .* diffs
    end

    return nothing
end

function calculate_virial!(virial_buff, cv::CalcRMSD, coords, forces, atoms, boundary; kwargs...)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    
    # RMSD with centering is translationally invariant.
    # We use the centroid of the current configuration as the reference point.
    # sum/length rather than mean(): mean() triggers scalar indexing (via
    # first()) when c_used is a @view of a CuArray
    com = sum(c_used) / length(c_used)

    # Accumulate sum( (r_i - r_centroid) * F_i^T )
    r_ic_all = vector.((com,), c_used, (boundary,))
    virial_buff .+= sum(r_ic_all .* transpose.(f_used))
end

"""
    CalcTorsion(atom_inds::AbstractVector{Int}=[], correction=:pbc, has_virial::Bool=true;
                gradient_singularity_tol=1e-6)

A collective variable that calculates the torsion angle (dihedral) defined by four atoms.

The angle is defined by the intersection of the planes formed by atoms (i, j, k) and (j, k, l), where the indices are given by `atom_inds`.
The torsion gradient is regularized near collinear geometries using
`gradient_singularity_tol`, a dimensionless relative tolerance applied to the
bond-vector norms.

# Fields
- `atom_inds::AbstractVector{Int}`: The indices of the four atoms (i, j, k, l) defining the torsion.
- `correction::Symbol`: The method used to handle periodic boundary conditions. Defaults to `:pbc`.
    `:pbc` runs fully on the GPU for GPU-resident `System`s, using a GPU-native bonded-topology
    traversal.
- `has_virial::Bool`: Whether the virial contribution should be calculated for this collective variable. Defaults to `true`.
- `gradient_singularity_tol::Float64`: Relative tolerance used to cap torsion gradients near collinear geometries.
"""
struct CalcTorsion
    atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool
    gradient_singularity_tol::Float64

    function CalcTorsion(atom_inds=[], correction=:pbc, has_virial=true;
                         gradient_singularity_tol=1e-6)
        check_correction_arg(correction)
        tol = Float64(gradient_singularity_tol)
        if !isfinite(tol) || tol <= 0
            throw(ArgumentError("gradient_singularity_tol must be finite and positive, " *
                                "got $gradient_singularity_tol"))
        end
        return new(atom_inds, correction, has_virial, tol)
    end
end

function calculate_cv(cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    buff = similar(coords, FT, 1)
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
    return only(from_device(buff))
end

function calculate_cv!(cv::CalcTorsion, coords, atoms, boundary, buff, args...; kwargs...)
    pts = from_device(coords[cv.atom_inds])
    buff .= torsion_angle(pts[1], pts[2], pts[3], pts[4], boundary)
    return nothing
end

# Single-thread GPU kernel avoids the from_device host sync the generic method above pays on
# every call (`coords[cv.atom_inds]` there is a full device->host sync).
@kernel inbounds=true function torsion_cv_kernel!(d_buf, @Const(coords), i, j, k, l, boundary)
    idx = @index(Global, Linear)
    if idx == 1
        d_buf[1] = torsion_angle(coords[i], coords[j], coords[k], coords[l], boundary)
    end
end

function calculate_cv!(cv::CalcTorsion, coords::AbstractGPUArray, atoms, boundary,
                       buff::AbstractGPUArray, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    backend = get_backend(coords)
    kernel! = torsion_cv_kernel!(backend, 1)
    kernel!(buff, coords, i, j, k, l, boundary; ndrange=1)
    return nothing
end

# Computes the analytical gradient of the torsion (dihedral) angle defined by four atoms.
#
# Mathematics:
# Let the four atoms be i, j, k, l. Define bond vectors: 
# b_1 = r_j - r_i,  b_2 = r_k - r_j,  b_3 = r_l - r_k.
# Define normal vectors to the planes: 
# m = b_1 x b_2,    n = b_2 x b_3. 
# 
# The gradients are evaluated via the chain rule on:
# ϕ = atan2( |b_2|(b_1 · n), m · n )
#
# This yields:
# ∇_{r_i} ϕ =   (|b_2| / |m|^2) * m
# ∇_{r_l} ϕ = - (|b_2| / |n|^2) * n
# ∇_{r_j} ϕ = - (1 + (b_1 · b_2)/|b_2|^2) * ∇_{r_i} ϕ + ((b_2 · b_3)/|b_2|^2) * ∇_{r_l} ϕ
# ∇_{r_k} ϕ =   ((b_1 · b_2)/|b_2|^2) * ∇_{r_i} ϕ - (1 + (b_2 · b_3)/|b_2|^2) * ∇_{r_l} ϕ
function check_torsion_bond_norm(norm_value, label::AbstractString)
    if !isfinite(ustrip(norm_value)) || norm_value <= zero(norm_value)
        throw(ArgumentError("CalcTorsion cannot compute a finite gradient because $(label) " *
                            "has non-positive or non-finite length ($(norm_value))"))
    end
    return norm_value
end

function cv_gradient(cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    grad = ustrip_vec.(zero(coords)) / oneunit(eltype(eltype(coords)))
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    d_buf = similar(coords, FT, 1)
    cv_gradient!(grad, d_buf, cv, coords, atoms, boundary, args...; kwargs...)
    return grad, only(from_device(d_buf))
end

function cv_gradient!(grad, d_buf, cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    pts = from_device(coords[[i, j, k, l]])
    ri, rj, rk, rl = pts[1], pts[2], pts[3], pts[4]

    b1 = vector(ri, rj, boundary)
    b2 = vector(rj, rk, boundary)
    b3 = vector(rk, rl, boundary)

    m = cross(b1, b2)
    n = cross(b2, b3)

    b1_norm = check_torsion_bond_norm(norm(b1), "bond i-j")
    b2_norm = check_torsion_bond_norm(norm(b2), "bond j-k")
    b3_norm = check_torsion_bond_norm(norm(b3), "bond k-l")
    FT = typeof(float(ustrip(b2_norm)))
    tol = FT(cv.gradient_singularity_tol)
    length_scale = max(b1_norm, b2_norm, b3_norm)
    norm_floor = tol * length_scale
    b1_norm_eff = max(b1_norm, norm_floor)
    b2_norm_eff = max(b2_norm, norm_floor)
    b3_norm_eff = max(b3_norm, norm_floor)

    m_sq = sum(abs2, m)
    n_sq = sum(abs2, n)
    b2_sq = b2_norm^2
    m_sq_eff = max(m_sq, (tol * b1_norm_eff * b2_norm_eff)^2)
    n_sq_eff = max(n_sq, (tol * b2_norm_eff * b3_norm_eff)^2)
    b2_sq_eff = max(b2_sq, b2_norm_eff^2)

    phi = torsion_angle(ri, rj, rk, rl, boundary)
    d_buf .= phi

    grad_i =  (b2_norm_eff / m_sq_eff) * m
    grad_l = -(b2_norm_eff / n_sq_eff) * n

    b1_dot_b2 = dot(b1, b2)
    b3_dot_b2 = dot(b3, b2)

    grad_j = -(1 + b1_dot_b2 / b2_sq_eff) * grad_i + (b3_dot_b2 / b2_sq_eff) * grad_l
    grad_k = (b1_dot_b2 / b2_sq_eff) * grad_i - (1 + b3_dot_b2 / b2_sq_eff) * grad_l

    grad[[i, j, k, l]] = -[grad_i, grad_j, grad_k, grad_l]

    return nothing
end

# check_torsion_bond_norm's CPU throw has no GPU equivalent (kernels can't throw catchable
# exceptions), so the degenerate case writes NaN into the affected atoms' gradient instead --
# still fails loud, via check_bias_finite's existing non-finite-gradient check (bias.jl).
@kernel inbounds=true function torsion_cv_gradient_kernel!(grad, d_buf, @Const(coords),
                                                            i, j, k, l, boundary, tol)
    idx = @index(Global, Linear)
    if idx == 1
        ri, rj, rk, rl = coords[i], coords[j], coords[k], coords[l]
        b1 = vector(ri, rj, boundary)
        b2 = vector(rj, rk, boundary)
        b3 = vector(rk, rl, boundary)
        m = cross(b1, b2)
        n = cross(b2, b3)
        b1n, b2n, b3n = norm(b1), norm(b2), norm(b3)
        d_buf[1] = torsion_angle(ri, rj, rk, rl, boundary)

        degenerate = !isfinite(ustrip(b1n)) || b1n <= zero(b1n) ||
                     !isfinite(ustrip(b2n)) || b2n <= zero(b2n) ||
                     !isfinite(ustrip(b3n)) || b3n <= zero(b3n)
        if degenerate
            nan_s = NaN / oneunit(b2n)   # same unit-attaching idiom as `zero(r_ij) / oneunit(d)` above
            nan_v = SVector(nan_s, nan_s, nan_s)
            grad[i] = nan_v; grad[j] = nan_v; grad[k] = nan_v; grad[l] = nan_v
        else
            length_scale = max(b1n, b2n, b3n)
            norm_floor = tol * length_scale
            b1n_eff = max(b1n, norm_floor)
            b2n_eff = max(b2n, norm_floor)
            b3n_eff = max(b3n, norm_floor)

            m_sq_eff = max(sum(abs2, m), (tol * b1n_eff * b2n_eff)^2)
            n_sq_eff = max(sum(abs2, n), (tol * b2n_eff * b3n_eff)^2)
            b2_sq_eff = max(b2n^2, b2n_eff^2)

            grad_i =  (b2n_eff / m_sq_eff) * m
            grad_l = -(b2n_eff / n_sq_eff) * n

            b1_dot_b2 = dot(b1, b2)
            b3_dot_b2 = dot(b3, b2)

            grad_j = -(1 + b1_dot_b2 / b2_sq_eff) * grad_i + (b3_dot_b2 / b2_sq_eff) * grad_l
            grad_k =  (b1_dot_b2 / b2_sq_eff) * grad_i - (1 + b3_dot_b2 / b2_sq_eff) * grad_l

            grad[i] = -grad_i; grad[j] = -grad_j; grad[k] = -grad_k; grad[l] = -grad_l
        end
    end
end

function cv_gradient!(grad::AbstractGPUArray, d_buf::AbstractGPUArray, cv::CalcTorsion,
                      coords::AbstractGPUArray, atoms, boundary, args...; kwargs...)
    i, j, k, l = cv.atom_inds
    FT = typeof(float(ustrip(oneunit(eltype(eltype(coords))))))
    tol = FT(cv.gradient_singularity_tol)
    backend = get_backend(coords)
    kernel! = torsion_cv_gradient_kernel!(backend, 1)
    kernel!(grad, d_buf, coords, i, j, k, l, boundary, tol; ndrange=1)
    return nothing
end

function calculate_virial!(virial_buff, cv::CalcTorsion, coords, forces, atoms, boundary; kwargs...)
    ids = cv.atom_inds
    pts = from_device(coords[ids])
    fs = from_device(forces[ids])
    c1, c2, c3, c4 = pts[1], pts[2], pts[3], pts[4]
    f1, f3, f4 = fs[1], fs[3], fs[4]
    r_ji = vector(c2, c1, boundary) # r_i - r_j
    r_jk = vector(c2, c3, boundary) # r_k - r_j
    r_jl = vector(c2, c4, boundary) # r_l - r_j

    virial_buff .+= r_ji * transpose(f1) +
                    r_jk * transpose(f3) +
                    r_jl * transpose(f4)
end

# --------------------------------------------------------------
# Persistent-buffer support for BiasPotential (src/bias/bias.jl).

# Trait gating BiasPotential's persistent-buffer path to CV types with a buffer-writing
# calculate_cv!/cv_gradient!; a custom CV that only implements calculate_cv falls back to the
# generic Enzyme-AD cv_gradient (ext/MollyEnzymeExt.jl), which has no buffer-writing equivalent.
uses_builtin_cv_gradient!(::CalcDist) = true
uses_builtin_cv_gradient!(::CalcRg) = true
uses_builtin_cv_gradient!(::CalcRMSD) = true
uses_builtin_cv_gradient!(::CalcTorsion) = true
uses_builtin_cv_gradient!(::Any) = false

# Buffer-shape helpers, deduplicating the grad/d_buf allocation pattern used by every allocating
# cv_gradient/calculate_cv wrapper above and by BiasPotential's lazy buffer init (bias.jl).
zero_cv_grad_buffer(cv, coords)  = ustrip_vec.(zero(coords))
zero_cv_value_buffer(cv, coords) = similar(coords, eltype(eltype(coords)), 1)
# CalcTorsion's CV value/gradient are unitless (an angle), unlike the other (length-valued) types.
zero_cv_grad_buffer(cv::CalcTorsion, coords) =
    ustrip_vec.(zero(coords)) / oneunit(eltype(eltype(coords)))
zero_cv_value_buffer(cv::CalcTorsion, coords) =
    similar(coords, typeof(float(ustrip(oneunit(eltype(eltype(coords)))))), 1)
zero_cv_gradient_buffers(cv, coords) = (zero_cv_grad_buffer(cv, coords), zero_cv_value_buffer(cv, coords))

"""
    calculate_cv_buffered!(cv, coords, atoms, boundary, buff, args...; kwargs...)

Uniform-signature wrapper around `calculate_cv!`, whose positional-argument prefix before `buff`
varies by CV type (`CalcRg` omits `boundary`; `CalcRMSD` omits `atoms`/`boundary`). Lets callers
that invoke `calculate_cv!` generically across CV types use one fixed call signature.
"""
calculate_cv_buffered!(cv::CalcRMSD, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, buff; kwargs...)
calculate_cv_buffered!(cv::CalcRg, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, atoms, buff; kwargs...)
calculate_cv_buffered!(cv, coords, atoms, boundary, buff, args...; kwargs...) =
    calculate_cv!(cv, coords, atoms, boundary, buff, args...; kwargs...)
