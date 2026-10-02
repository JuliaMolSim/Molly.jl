# Neighbor finders

export
    use_neighbors,
    NoNeighborFinder,
    find_neighbors,
    GPUNeighborFinder,
    DistanceNeighborFinder,
    TreeNeighborFinder,
    CellListMapNeighborFinder

"""
    use_neighbors(inter)

Whether a pairwise interaction uses the neighbor list, default `false`.

Custom pairwise interactions can define a method for this function.
For built-in interactions such as [`LennardJones`](@ref) this function accesses
the `use_neighbors` field of the struct.
"""
use_neighbors(inter) = false

function check_neighbor_matrices(eligible, special)
    if !isnothing(eligible) && !isnothing(special) && size(eligible) != size(special)
        throw(ArgumentError("size of the eligible matrix $(size(eligible)) must be " *
                            "the same as the size of the special matrix $(size(special))"))
    end
    if !isnothing(eligible) && !is_symmetric_mask(eligible)
        throw(ArgumentError("eligible matrix is not symmetric"))
    end
    if !isnothing(special) && !is_symmetric_mask(special)
        throw(ArgumentError("special matrix is not symmetric"))
    end
end

is_symmetric_mask(m) = issymmetric(m)
# issymmetric on a BitMatrix builds A - A' as a dense Int matrix (8 bytes per atom pair)
is_symmetric_mask(m::BitMatrix) = (m == copy(transpose(m)))

# Number of atoms above which a warning is given when a neighbor finder is given a dense
#   eligible or special matrix
const dense_matrix_warn_n_atoms = 50_000

function check_dense_neighbor_matrix(matrix, name, strictness;
                                     warn_n_atoms::Integer=dense_matrix_warn_n_atoms)
    if matrix isa Union{DenseArray, BitArray} && size(matrix, 1) > warn_n_atoms
        err_str = "a dense $name matrix was given to a neighbor finder for " *
                  "$(size(matrix, 1)) atoms, which takes memory proportional to " *
                  "n_atoms^2 and limits the size of system that can be simulated, give " *
                  "n_atoms along with excluded_pairs and special_pairs instead to store " *
                  "the pairs sparsely"
        report_issue(err_str, strictness; error_type=ArgumentError)
    end
    return nothing
end

#=
The eligible and special matrices of a neighbor finder from its keyword arguments.
Either `n_atoms` is given with `excluded_pairs` and `special_pairs`, in which case sparse
    matrices are built with their lists stored as `array_type` (`Array` if `nothing`), or
    `eligible`, and optionally `special`, is given as a matrix, dense or a
    SparsePairMatrix, in which case the matrices are moved to `array_type` if that is not
    `nothing`.
A missing `special` matrix means that no pairs are special.
=#
function neighbor_matrices(eligible, special, n_atoms, excluded_pairs, special_pairs,
                           array_type, strictness)
    check_strictness(strictness)
    if !isnothing(array_type)
        check_neighbor_array_type(array_type)
    end
    # Checked before anything else, as the other checks can take a long time for a
    #   large dense matrix
    check_dense_neighbor_matrix(eligible, "eligible", strictness)
    check_dense_neighbor_matrix(special, "special", strictness)
    if isnothing(eligible)
        if isnothing(n_atoms)
            throw(ArgumentError("either n_atoms or eligible must be provided"))
        end
        pairs_array_type = something(array_type, Array)
        eligible_used = SparsePairMatrix(n_atoms, excluded_pairs; listed=false,
                                         array_type=pairs_array_type)
        if isnothing(special)
            special_used = SparsePairMatrix(n_atoms, special_pairs; listed=true,
                                            array_type=pairs_array_type)
        else
            if special_pairs !== ()
                throw(ArgumentError("special and special_pairs cannot both be provided"))
            end
            special_used = special
        end
    else
        if excluded_pairs !== () || special_pairs !== ()
            throw(ArgumentError("excluded_pairs and special_pairs cannot be used with " *
                                "eligible, give n_atoms instead of eligible"))
        end
        if !isnothing(n_atoms) && size(eligible) != (n_atoms, n_atoms)
            throw(ArgumentError("n_atoms is $n_atoms but the eligible matrix has size " *
                                "$(size(eligible))"))
        end
        eligible_used = eligible
        special_used = (isnothing(special) ? zero(eligible) : special)
    end
    if !isnothing(array_type)
        eligible_used = move_neighbor_matrix(eligible_used, array_type)
        special_used = move_neighbor_matrix(special_used, array_type)
    end
    check_neighbor_matrices(eligible_used, special_used)
    return eligible_used, special_used
end

function move_neighbor_matrix(matrix, AT)
    if AT <: AbstractGPUArray || neighbor_matrix_on_gpu(matrix)
        return to_device(matrix, AT)
    else
        return matrix
    end
end

function check_neighbor_array_type(AT)
    if !(AT isa Type && AT <: AbstractArray)
        throw(ArgumentError("array_type must be an array type such as Array or CuArray, " *
                            "found $AT"))
    end
    if !(to_device(Int32[], AT) isa AbstractVector{Int32})
        throw(ArgumentError("array_type must be able to store Int32 vectors, as the " *
                            "lists of pairs are stored with it, found $AT"))
    end
    return AT
end

"""
    NoNeighborFinder()

Placeholder neighbor finder that returns no neighbors.

When using this neighbor finder, ensure that [`use_neighbors`](@ref) for the interactions
returns `false`.
"""
struct NoNeighborFinder end

"""
    find_neighbors(system; n_threads=Threads.nthreads())
    find_neighbors(system, neighbor_finder, current_neighbors=nothing, step_n=0,
                   force_recompute=false; n_threads=Threads.nthreads())

Obtain a list of close atoms in a [`System`](@ref).

Custom neighbor finders should implement this function.

For [`GPUNeighborFinder`](@ref), this returns `nothing`: the CUDA pairwise force
and energy kernels build and cache their interacting tile list internally from
the neighbor-finder metadata.
"""
find_neighbors(sys::System; kwargs...) = find_neighbors(sys, sys.neighbor_finder; kwargs...)

find_neighbors(sys::System, nf::NoNeighborFinder, args...; kwargs...) = nothing

#=
    uses_gpu_neighbor_finder(AT)

Indicate whether an array type `AT` is compatible with [`GPUNeighborFinder`](@ref).

Custom GPU array types should define a method for this function that returns `true`
if they are supported. By default, this returns `false`.
=#
uses_gpu_neighbor_finder(AT) = false

"""
    GPUNeighborFinder(; n_atoms, dist_cutoff, excluded_pairs=(), special_pairs=(),
                      n_steps=10, array_type, initialized=false, strictness=:warn)
    GPUNeighborFinder(; eligible, dist_cutoff, special=nothing, n_steps=10,
                      array_type=nothing, initialized=false, strictness=:warn)

Neighbor finder for CUDA systems that uses Molly's tiled pairwise kernels.

`GPUNeighborFinder` does not materialize a conventional per-atom neighbor list.
Instead, the CUDA pairwise force and energy paths reorder atoms on a Morton
curve and build a compact list of interacting 32x32 tiles directly on the device.
The exclusions and special pairs are read from per-atom sparse lists inside the
tiles that contain them, so the memory used grows linearly with the number of
atoms.
For that reason, [`find_neighbors`](@ref) returns `nothing` for this neighbor
finder.

This is the recommended neighbor finder for `CuArray` systems.
The exceptions are stored in the `eligible` and `special` fields as
[`SparsePairMatrix`](@ref)s on the device. Updating the exception pairs should reset
`initialized` so the cached tile data is rebuilt on the next GPU force or energy evaluation.

# Keyword arguments
- `n_atoms`: the number of atoms in the system.
- `dist_cutoff`: the neighbor search distance used by the pairwise kernels.
  This should be the interaction cutoff distance plus a buffer distance, since
  the tile list is only refreshed every `n_steps` steps. The buffer should be
  larger than the distance an atom can move in that time.
- `excluded_pairs`: an iterable of `(i, j)` pairs, or a sparse matrix, of the pairs
  that are excluded from the pairwise interactions, for example bonded atoms.
- `special_pairs`: an iterable of `(i, j)` pairs, or a sparse matrix, of the pairs
  that use the special interaction path, for example 1-4 atoms.
- `n_steps`: the number of steps between Morton reorder and tile-list refresh
  passes.
- `array_type`: the array type used to store the excluded and special pairs on
  the GPU, for example `CuArray`. It is required unless `eligible` is given on
  the GPU, in which case its array type is used.
- `eligible`, `special`: eligible and special matrices can be given instead of
  `n_atoms`, `excluded_pairs` and `special_pairs`, see
  [`DistanceNeighborFinder`](@ref). They are converted once at construction into
  sparse matrices and are not retained.
- `initialized`: whether the current sparse-exception preprocessing state can be
  reused.
- `strictness=:warn`: determines behavior when a dense `eligible` or `special`
  matrix is given for more than 50,000 atoms, options are `:warn` to emit
  warnings, `:nowarn` to suppress warnings or `:error` to error.
"""
mutable struct GPUNeighborFinder{B, D, D2, E}
    n_atoms::B
    dist_cutoff::D
    dist_cutoff_2::D2
    n_steps::Int
    initialized::Bool
    cache_generation::UInt64
    eligible::E
    special::E
end

#=
The array type used to store the sparse lists of a GPUNeighborFinder: `AT` if it is
    given, otherwise the array type of the eligible matrix if that is on the GPU.
=#
function gpu_neighbor_array_type(eligible, AT)
    if isnothing(AT)
        if eligible isa AbstractGPUArray
            return array_type(eligible)
        elseif eligible isa SparsePairMatrix && neighbor_matrix_on_gpu(eligible)
            return array_type(eligible.starts)
        end
        throw(ArgumentError("array_type must be given, for example array_type=CuArray, " *
                            "unless the eligible matrix is given on the GPU"))
    end
    return check_neighbor_array_type(AT)
end

#=
    dense_masks_to_pair_lists(eligible, special)

Convert eligible and special matrices, dense or sparse, into sorted lists of `(i, j)`,
`i < j`, excluded and special pairs.
=#
function dense_masks_to_pair_lists(eligible, special)
    return ineligible_pairs(eligible), true_pairs(special)
end

function neighbor_finder_masks(nf::NoNeighborFinder, n_atoms::Integer)
    eligible = trues(n_atoms, n_atoms)
    special = falses(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    return eligible, special
end

neighbor_finder_masks(nf, ::Integer) = neighbor_finder_masks(nf)

#=
    update_sparse_pairs!(nf, excluded_pairs, special_pairs)

Replace the existing exception lists in a [`GPUNeighborFinder`](@ref) with new ones.

This normalizes the `excluded_pairs` and `special_pairs`, uploads them to the
device, and resets the neighbor finder's `initialized` flag so that the cached
GPU tile data is rebuilt.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `excluded_pairs`: an iterable of `(i, j)` pairs that should be excluded.
- `special_pairs`: an iterable of `(i, j)` pairs that should use the special interaction path.
=#
function update_sparse_pairs!(nf::GPUNeighborFinder, excluded_pairs, special_pairs)
    ET = typeof(nf.eligible.starts)
    eligible = SparsePairMatrix(nf.n_atoms, excluded_pairs; listed=false, array_type=ET)
    special = SparsePairMatrix(nf.n_atoms, special_pairs; listed=true, array_type=ET)
    nf.eligible, nf.special = eligible, special
    nf.initialized = false
    nf.cache_generation += 0x0000000000000001
    return nf
end

#=
    append_excluded_pairs!(nf, pairs)

Append new excluded pairs to the existing exclusions in a [`GPUNeighborFinder`](@ref).

The `initialized` state will be reset.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `pairs`: an iterable of `(i, j)` pairs to add to the exclusions list.
=#
function append_excluded_pairs!(nf::GPUNeighborFinder, pairs)
    add_listed_pairs!(nf.eligible, pairs)
    nf.initialized = false
    nf.cache_generation += 0x0000000000000001
    return nf
end

function GPUNeighborFinder(;
                            n_atoms=nothing,
                            dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            n_steps=10,
                            array_type=nothing,
                            eligible=nothing,
                            special=nothing,
                            initialized=false,
                            strictness=default_strictness())
    if isnothing(eligible) && isnothing(n_atoms)
        throw(ArgumentError("either n_atoms or eligible must be provided"))
    end
    AT = gpu_neighbor_array_type(eligible, array_type)
    # The matrices are left where they are, since dense ones are only read once here
    eligible_given, special_given = neighbor_matrices(eligible, special, n_atoms,
                        excluded_pairs, special_pairs, nothing, strictness)
    # The kernels read sparse lists on the device, dense matrices are converted
    n_atoms_used = size(eligible_given, 1)
    eligible_sparse = sparse_eligible(eligible_given, n_atoms_used, AT)
    special_sparse = sparse_special(special_given, n_atoms_used, AT)
    dist_cutoff_2 = dist_cutoff^2
    return GPUNeighborFinder{Int, typeof(dist_cutoff), typeof(dist_cutoff_2), typeof(eligible_sparse)}(
                Int(n_atoms_used), dist_cutoff, dist_cutoff_2, n_steps, initialized, 0,
                eligible_sparse, special_sparse)
end

# The interacting tile list is constructed within the CUDA pairwise kernels.
find_neighbors(sys::System, nf::GPUNeighborFinder, args...; kwargs...) = nothing

# Mark neighbor data cached in `buffers` as stale so that it is rebuilt on the next force
#   or energy evaluation
# Neighbor finders that return a neighbor list from `find_neighbors` do not cache
#   anything in the buffers, so this does nothing for them
invalidate_cached_neighbors!(buffers, neighbor_finder) = buffers

function invalidate_cached_neighbors!(buffers::BuffersGPU, nf::GPUNeighborFinder)
    buffers.step_n_preprocessed = -1
    return buffers
end

"""
    DistanceNeighborFinder(; n_atoms, dist_cutoff, excluded_pairs=(), special_pairs=(),
                           n_steps=10, array_type=nothing, strictness=:warn)
    DistanceNeighborFinder(; eligible, dist_cutoff, special=nothing, n_steps=10,
                           array_type=nothing, strictness=:warn)

Find close atoms by distance.

This is the recommended neighbor finder on non-NVIDIA GPUs.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.

The pairs of atoms that can interact are described by `n_atoms`, the number of atoms,
along with `excluded_pairs`, the pairs excluded from the pairwise interactions such as
bonded atoms, and `special_pairs`, the pairs that use the special interaction path such
as 1-4 atoms.
These are iterables of `(i, j)` pairs or sparse matrices.
They are stored as [`SparsePairMatrix`](@ref)s, so the memory used is proportional to the
number of pairs.
`array_type` is the array type used to store them, `Array` by default, and should match
the [`System`](@ref) that the neighbor finder is used with, for example `CuArray` for a
system on a NVIDIA GPU.

Alternatively, `eligible` can be given as a symmetric `n_atoms` x `n_atoms` Boolean
matrix that is `true` for the pairs that can interact, along with `special` as a matrix
of the same size that is `true` for the special pairs, or `nothing` if no pairs are
special.
If `array_type` is given the matrices are moved to it.
Dense matrices take memory proportional to `n_atoms^2`, so they are only suitable for
small systems.
`strictness` determines the behavior when a dense matrix is given for more than 50,000
atoms, options are `:warn` to emit a warning, `:nowarn` to suppress it or `:error` to
error.
"""
struct DistanceNeighborFinder{B, D, S}
    eligible::B
    dist_cutoff::D
    special::S
    n_steps::Int
end

function DistanceNeighborFinder(;
                                n_atoms=nothing,
                                dist_cutoff,
                                excluded_pairs=(),
                                special_pairs=(),
                                n_steps=10,
                                array_type=nothing,
                                eligible=nothing,
                                special=nothing,
                                strictness=default_strictness())
    eligible_used, special_used = neighbor_matrices(eligible, special, n_atoms,
                                    excluded_pairs, special_pairs, array_type, strictness)
    return DistanceNeighborFinder{typeof(eligible_used), typeof(dist_cutoff),
                                  typeof(special_used)}(
                eligible_used, dist_cutoff, special_used, n_steps)
end

#=
The neighbor search for `DistanceNeighborFinder` is done in two passes on both CPU and
    GPU, which avoids growing intermediate lists and lets the output be written exactly
    once into an array of the right size.
The first pass records whether each candidate pair is a neighbor as one bit of a mask
    word and counts the bits set in that word, the counts are turned into write offsets
    with a prefix sum, and the second pass expands the mask words into the neighbor list.
Splitting the distance test off from the list building also lets the distance loop be
    vectorized on the CPU, which it cannot be when it contains a `push!`.
=#

# Number of pair flags packed into one CPU mask word
const n_pairs_per_mask_cpu = 64

# Multiplier that gathers the low bit of each of 8 bytes into the top byte of the product
const byte_flags_to_bits = 0x0102040810204080

#=
Record in `flags[j]` whether atom j is within the cutoff of the atom at `ci` and eligible
Written as a separate loop from the list building so that it can be vectorized
=#
@inline function neighbor_flags!(flags, coords_dims, ci, boundary, sqdist_cutoff, eligible_i,
                                 n_j, ::Val{D}) where D
    @inbounds for j in 1:n_j
        cj = SVector{D}(ntuple(d -> @inbounds(coords_dims[d][j]), Val(D)))
        r2 = sum(abs2, vector(ci, cj, boundary))
        flags[j] = (r2 <= sqdist_cutoff) & eligible_i[j]
    end
    return flags
end

#=
The column of the eligible matrix used in the vectorized distance loop.
A SparsePairMatrix is looked up by scanning a list, which would stop the loop being
    vectorized, so every pair is treated as eligible in the loop and
    `apply_sparse_eligible!` corrects the flags of the listed pairs afterwards.
=#
eligible_column(eligible::AbstractMatrix, i) = @view eligible[:, i]
eligible_column(eligible::SparsePairMatrix, i) = Fill(true, size(eligible, 1))

apply_sparse_eligible!(flags, eligible::AbstractMatrix, i) = flags

# Apply the eligibility of the pairs (j, i), j < i, listed in a SparsePairMatrix to flags
#   set from the distances alone
function apply_sparse_eligible!(flags, eligible::SparsePairMatrix, i)
    starts, partners = eligible.starts, eligible.partners
    @inbounds k_start, k_end = starts[i], starts[i + 1]
    if eligible.listed
        # Only the listed pairs are eligible
        k = k_start
        @inbounds for j in 1:(i - 1)
            if k < k_end && partners[k] == j
                k += one(k)
            else
                flags[j] = 0x00
            end
        end
    else
        # The listed pairs are not eligible
        @inbounds for k in k_start:(k_end - one(k_end))
            j = partners[k]
            j >= i && break
            flags[j] = 0x00
        end
    end
    return flags
end

#=
Pack the first `n_j` 0/1 bytes in `flags` into mask words starting at `mask_offset`
`flag_words` is `flags` viewed 8 bytes at a time, which makes the packing 8 times cheaper
Any bits of the last word past `n_j` are zeroed, so the flags there do not have to be
Returns the number of bits set, i.e. the number of neighbors found for the atom
=#
@inline function pack_neighbor_masks!(masks, flag_words, mask_offset, n_j)
    n_words = cld(n_j, n_pairs_per_mask_cpu)
    n_set = 0
    @inbounds for w in 1:n_words
        word_start = (w - 1) << 3
        mask = zero(UInt64)
        for b in 0:7
            byte_flags = flag_words[word_start + b + 1]
            mask |= UInt64((byte_flags * byte_flags_to_bits) >> 56) << (b << 3)
        end
        n_bits = n_j - ((w - 1) << 6)
        if n_bits < n_pairs_per_mask_cpu
            mask &= (one(UInt64) << n_bits) - one(UInt64)
        end
        masks[mask_offset + w] = mask
        n_set += count_ones(mask)
    end
    return n_set
end

# Expand the mask words for atom i into `neighbors_list`, starting after index `list_offset`
@inline function expand_neighbor_masks!(neighbors_list, masks, mask_offset, n_j, i,
                                        special::AbstractMatrix, list_offset)
    special_i = @view special[:, i]
    ni = list_offset
    @inbounds for w in 1:cld(n_j, n_pairs_per_mask_cpu)
        mask = masks[mask_offset + w]
        while !iszero(mask)
            j = ((w - 1) << 6) + trailing_zeros(mask) + 1
            mask &= mask - one(mask)
            ni += 1
            neighbors_list[ni] = (Int32(i), Int32(j), special_i[j])
        end
    end
    return ni
end

# The neighbors j are found in ascending order, as are the partners of atom i in a
#   SparsePairMatrix, so the partner list is walked alongside the neighbors
@inline function expand_neighbor_masks!(neighbors_list, masks, mask_offset, n_j, i,
                                        special::SparsePairMatrix, list_offset)
    partners, listed_value = special.partners, special.listed
    @inbounds k, k_end = special.starts[i], special.starts[i + 1]
    ni = list_offset
    @inbounds for w in 1:cld(n_j, n_pairs_per_mask_cpu)
        mask = masks[mask_offset + w]
        while !iszero(mask)
            j = ((w - 1) << 6) + trailing_zeros(mask) + 1
            mask &= mask - one(mask)
            while k < k_end && partners[k] < j
                k += one(k)
            end
            listed = (k < k_end && partners[k] == j)
            ni += 1
            neighbors_list[ni] = (Int32(i), Int32(j), listed == listed_value)
        end
    end
    return ni
end

function find_neighbors(sys::System{D},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where D
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    n_atoms = length(sys)
    sqdist_cutoff = nf.dist_cutoff ^ 2
    # The coordinates are copied to one array per dimension since the strided reads of an
    #   array of static vectors stop the distance loop being vectorized
    coords_dims = ntuple(d -> [c[d] for c in sys.coords], Val(D))

    # Mask words are packed by atom, with the words for atom i starting at mask_starts[i]
    mask_starts = Vector{Int}(undef, n_atoms)
    n_mask_words = 0
    @inbounds for i in 1:n_atoms
        mask_starts[i] = n_mask_words
        n_mask_words += cld(i - 1, n_pairs_per_mask_cpu)
    end
    masks = Vector{UInt64}(undef, n_mask_words)
    n_neighbors_atom = Vector{Int}(undef, n_atoms)
    # Rounded up to a whole number of mask words so that the flags can be viewed as words
    n_flags = cld(n_atoms, n_pairs_per_mask_cpu) * n_pairs_per_mask_cpu
    flags_threads = [zeros(UInt8, n_flags) for _ in 1:n_threads]

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        flags = flags_threads[chunk_i]
        flag_words = reinterpret(UInt64, flags)
        @inbounds for i in chunk_i:n_threads:n_atoms
            neighbor_flags!(flags, coords_dims, sys.coords[i], sys.boundary, sqdist_cutoff,
                            eligible_column(nf.eligible, i), i - 1, Val(D))
            apply_sparse_eligible!(flags, nf.eligible, i)
            n_neighbors_atom[i] = pack_neighbor_masks!(masks, flag_words, mask_starts[i], i - 1)
        end
    end

    # Exclusive prefix sum of the per-atom neighbor counts gives the write offsets
    list_starts = Vector{Int}(undef, n_atoms)
    n_neighbors = 0
    @inbounds for i in 1:n_atoms
        list_starts[i] = n_neighbors
        n_neighbors += n_neighbors_atom[i]
    end
    neighbors_list = Vector{Tuple{Int32, Int32, Bool}}(undef, n_neighbors)

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        @inbounds for i in chunk_i:n_threads:n_atoms
            expand_neighbor_masks!(neighbors_list, masks, mask_starts[i], i - 1, i,
                                   nf.special, list_starts[i])
        end
    end

    return NeighborList(n_neighbors, neighbors_list)
end

function gpu_threads_dnf(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_DISTANCENF", "512"))
    return n_threads_gpu
end

const n_pairs_per_mask_gpu = 32 # n pair flags packed into one GPU mask word
const n_masks_per_group_gpu = 32 # n mask words filled by one group of consecutive GPU threads
const n_pairs_per_group_gpu = n_pairs_per_mask_gpu * n_masks_per_group_gpu

#=
Map a one-based index over the n_atoms * (n_atoms - 1) / 2 pairs to the pair (i, j), i < j,
    running down the columns of the pair triangle so that consecutive indices give
    consecutive i for a fixed j.
This is the transpose of `pair_index` and is used by the GPU neighbor finder kernels, where
    it makes the `eligible[i, j]` and `coords[i]` reads of neighboring threads coalesced.
As in `pair_index` the square root is taken in Float32 since Metal GPUs do not support
    Float64, so the initial estimate of j is corrected below using exact integer arithmetic.
=#
@inline function pair_index_col(n_atoms::Integer, ind::Integer)
    n, kz = promote(n_atoms, ind - one(ind))
    T = typeof(n)
    # Column j holds j - 1 pairs, so jz = j - 1 is the largest value with
    #   jz * (jz - 1) / 2 <= kz
    jz = unsafe_trunc(T, (sqrt(Float32(8 * kz + 1)) + 1.0f0) / 2)
    jz = min(max(jz, one(T)), n - one(T))
    while jz > one(T) && (jz * (jz - one(T))) ÷ T(2) > kz
        jz -= one(T)
    end
    while jz < n - one(T) && ((jz + one(T)) * jz) ÷ T(2) <= kz
        jz += one(T)
    end
    i = kz - (jz * (jz - one(T))) ÷ T(2) + one(T)
    j = jz + one(T)
    return i, j
end

#=
Map bit `bit_i` (zero-based) of mask word `word_i` to the index of the pair it records.
Consecutive threads in a group take consecutive pairs at each step so that the coordinate
    reads are coalesced, meaning that consecutive bits of a mask word are
    `n_masks_per_group_gpu` pairs apart.
=#
@inline function mask_bit_pair_index(word_i, bit_i)
    group_i, word_in_group = divrem(word_i - 1, n_masks_per_group_gpu)
    return group_i * n_pairs_per_group_gpu + bit_i * n_masks_per_group_gpu + word_in_group + 1
end

# `eligible` is a dense matrix or the form of a SparsePairMatrix from `kernel_matrix`
# It is only read for pairs within the cutoff, which are a small fraction of all pairs
@kernel inbounds=true function distance_neighbor_finder_mask_kernel!(masks, counts, @Const(coords),
                                        eligible, boundary, sq_dist_cutoff, n_inters)
    n_atoms = length(coords)
    word_i = @index(Global, Linear)

    if word_i <= length(masks)
        mask = zero(UInt32)
        for bit_i in 0:(n_pairs_per_mask_gpu - 1)
            inter_i = mask_bit_pair_index(word_i, bit_i)
            if inter_i <= n_inters
                i, j = pair_index_col(n_atoms, inter_i)
                dr = vector(coords[i], coords[j], boundary)
                r2 = sum(abs2, dr)
                if r2 <= sq_dist_cutoff && pair_value(eligible, i, j)
                    mask |= (one(UInt32) << bit_i)
                end
            end
        end
        masks[word_i] = mask
        counts[word_i] = Int32(count_ones(mask))
    end
end

@kernel inbounds=true function distance_neighbor_finder_fill_kernel!(neighbors_list, @Const(masks),
                                        @Const(list_ends), special, n_atoms)
    word_i = @index(Global, Linear)

    if word_i <= length(masks)
        mask = masks[word_i]
        if !iszero(mask)
            ni = list_ends[word_i] - Int32(count_ones(mask))
            while !iszero(mask)
                bit_i = trailing_zeros(mask)
                mask &= mask - one(mask)
                i, j = pair_index_col(n_atoms, mask_bit_pair_index(word_i, bit_i))
                ni += Int32(1)
                neighbors_list[ni] = (Int32(j), Int32(i), pair_value(special, j, i))
            end
        end
    end
end

function find_neighbors(sys::System{D, AT},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...) where {D, AT <: AbstractGPUArray}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    n_inters = n_atoms_to_n_pairs(length(sys))
    if iszero(n_inters)
        return NeighborList(0, similar(sys.coords, Tuple{Int32, Int32, Bool}, 0))
    end
    n_threads_gpu = gpu_threads_dnf(n_inters)
    backend = get_backend(sys.coords)

    # Rounded up to a whole number of groups so that every pair is covered by a mask bit
    n_masks = cld(n_inters, n_pairs_per_group_gpu) * n_masks_per_group_gpu
    masks = similar(sys.coords, UInt32, n_masks)
    counts = similar(sys.coords, Int32, n_masks)

    mask_kernel! = distance_neighbor_finder_mask_kernel!(backend, n_threads_gpu)
    mask_kernel!(masks, counts, sys.coords, kernel_matrix(nf.eligible), sys.boundary,
                 nf.dist_cutoff^2, n_inters; ndrange=n_masks)

    # The inclusive prefix sum of the per-word neighbor counts gives the index one past the
    #   last neighbor written by each mask word, from which the fill kernel subtracts its
    #   own count to get its write offset
    # The inclusive version is used because AcceleratedKernels.accumulate! with
    #   inclusive=false gives the wrong result past the first block
    AcceleratedKernels.accumulate!(+, counts, backend; init=Int32(0))
    n_neighbors = Int(only(Array(@view counts[n_masks:n_masks])))
    neighbors_list = similar(sys.coords, Tuple{Int32, Int32, Bool}, n_neighbors)

    if n_neighbors > 0
        fill_kernel! = distance_neighbor_finder_fill_kernel!(backend, n_threads_gpu)
        fill_kernel!(neighbors_list, masks, counts, kernel_matrix(nf.special), length(sys);
                     ndrange=n_masks)
    end

    return NeighborList(n_neighbors, neighbors_list)
end

"""
    TreeNeighborFinder(; n_atoms, dist_cutoff, excluded_pairs=(), special_pairs=(),
                       n_steps=10, strictness=:warn)
    TreeNeighborFinder(; eligible, dist_cutoff, special=nothing, n_steps=10,
                       strictness=:warn)

Find close atoms by distance using a tree search.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
The pairs of atoms that can interact are given by `n_atoms`, `excluded_pairs` and
`special_pairs`, or by the `eligible` and `special` matrices, as for
[`DistanceNeighborFinder`](@ref).
The pairs are stored on the CPU.

Can not be used if one or more dimensions has infinite boundaries.
Can not be used with [`TriclinicBoundary`](@ref).
"""
struct TreeNeighborFinder{D, B, S}
    eligible::B
    dist_cutoff::D
    special::S
    n_steps::Int
end

# The neighbor finders that run on the CPU store dense matrices as a BitMatrix
cpu_neighbor_matrix(m::SparsePairMatrix) = from_device(m)
cpu_neighbor_matrix(m) = to_bitmatrix(from_device(m))

function TreeNeighborFinder(;
                            n_atoms=nothing,
                            dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            n_steps=10,
                            eligible=nothing,
                            special=nothing,
                            strictness=default_strictness())
    eligible_used, special_used = neighbor_matrices(eligible, special, n_atoms,
                                    excluded_pairs, special_pairs, nothing, strictness)
    return TreeNeighborFinder(cpu_neighbor_matrix(eligible_used), dist_cutoff,
                              cpu_neighbor_matrix(special_used), n_steps)
end

function find_neighbors(sys::System{<:Any, AT},
                        nf::TreeNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where AT
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    dist_unit = unit(first(first(sys.coords)))
    bv = ustrip.(dist_unit, sys.boundary)
    btree = BallTree(ustrip_vec.(sys.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)
    nl_threads = [Tuple{Int32, Int32, Bool}[] for i in 1:n_threads]
    eligible, special = kernel_matrix(nf.eligible), kernel_matrix(nf.special)

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(sys)
            ci = ustrip.(sys.coords[i])
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if i > j && pair_value(eligible, j, i)
                    push!(nl_threads[chunk_i], (Int32(i), Int32(j), pair_value(special, j, i)))
                end
            end
        end
    end

    neighbors_list = Tuple{Int32, Int32, Bool}[]
    for nl in nl_threads
        append!(neighbors_list, nl)
    end

    return NeighborList(length(neighbors_list), to_device(neighbors_list, AT))
end

"""
    CellListMapNeighborFinder(; n_atoms, dist_cutoff, boundary, excluded_pairs=(),
                                special_pairs=(), n_steps=10, x0=nothing,
                                number_of_batches=(0, 0), strictness=:warn)
    CellListMapNeighborFinder(; eligible, dist_cutoff, boundary, special=nothing,
                                n_steps=10, x0=nothing, number_of_batches=(0, 0),
                                strictness=:warn)

Find close atoms by distance using a cell list algorithm from CellListMap.jl.

This is the recommended neighbor finder on CPU.
`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
The pairs of atoms that can interact are given by `n_atoms`, `excluded_pairs` and
`special_pairs`, or by the `eligible` and `special` matrices, as for
[`DistanceNeighborFinder`](@ref).
The pairs are stored on the CPU.
`x0` are optional initial coordinates that improve the
first approximation of the cell list structure.
The number of dimensions `dims` is inferred from the boundary or `x0`, or assumed
to be 3 otherwise.

The `boundary` parameter is required, and must be an `AbstractBoundary`. 
Infinite boundaries are only accepted in all dimensions.

CellListMap.jl chooses how many batches to split the work over from the number of Julia
threads, so the `n_threads` argument to [`find_neighbors`](@ref) only selects between a
serial (`n_threads=1`) and a parallel run. `number_of_batches` can be given as a tuple of
the number of batches used to build the cell lists and to map over the pairs, which caps
the number of tasks used, with `(0, 0)`, the default, meaning to use the CellListMap.jl
heuristics.
"""
mutable struct CellListMapNeighborFinder{N, T, S, B, SP}
    eligible::B
    dist_cutoff::T
    special::SP
    n_steps::Int
    # The CellListMap.ParticleSystem object
    clm_particlesystem::S
end

function clm_unitcell_arg(b::Union{CubicBoundary, RectangularBoundary}) 
    uc = b.side_lengths
    D = size(uc, 1)
    if any(isinf.(uc))
        if all(isinf.(uc))
            return nothing, D
        else
            throw(ArgumentError("cannot use infinite boundaries in some, but not all, " *
                                "dimensions with CellListMapNeighborFinder"))
        end
    end
    return uc, D
end

function clm_unitcell_arg(b::TriclinicBoundary) 
    uc = hcat(b.basis_vectors...)
    D = size(uc, 1)
    return uc, D
end

# This function sets up the ParticleSystem structure for CellListMap. 
function CellListMapNeighborFinder(;
                                   n_atoms=nothing,
                                   dist_cutoff::T,
                                   boundary::AbstractBoundary,
                                   excluded_pairs=(),
                                   special_pairs=(),
                                   n_steps=10,
                                   x0=nothing,
                                   number_of_batches=(0, 0),
                                   eligible=nothing,
                                   special=nothing,
                                   strictness=default_strictness()) where T
    eligible_used, special_used = neighbor_matrices(eligible, special, n_atoms,
                                    excluded_pairs, special_pairs, nothing, strictness)
    eligible_cpu, special_cpu = cpu_neighbor_matrix(eligible_used), cpu_neighbor_matrix(special_used)
    # Obtain unit cell from boundary: If all boundaries are infinite, use `nothing`
    uc, D = clm_unitcell_arg(boundary)

    clm_system = CellListMap.ParticleSystem(;
        positions=isnothing(x0) ? SVector{D,T}[] : x0,
        cutoff=dist_cutoff,
        unitcell=uc,
        nbatches=number_of_batches,
        parallel=true,
        output=NeighborList(),
        output_name=:neighbors,
    )

    return CellListMapNeighborFinder{D, T, typeof(clm_system), typeof(eligible_cpu),
                                     typeof(special_cpu)}(
                eligible_cpu, dist_cutoff, special_cpu, n_steps, clm_system)
end

# Add a pair to the pair list
# If the buffer size is large enough update the element, otherwise push a new element
#   to `neighbors.list`
# `eligible` and `special` are dense matrices or the form of a SparsePairMatrix from
#   `kernel_matrix`
@inline function push_pair!(pair, neighbors::NeighborList, eligible, special)
    i, j = pair.i, pair.j
    @inbounds if pair_value(eligible, i, j)
        n = neighbors.n + 1
        neighbors.n = n
        list = neighbors.list
        element = (i, j, pair_value(special, i, j))
        if n > length(list)
            push!(list, element)
        else
            @inbounds list[n] = element
        end
    end
    return neighbors
end

# Parallelization interface for custom output of CellListMap
CellListMap.copy_output(nl::NeighborList) = NeighborList(nl.n, copy(nl.list)) 
CellListMap.reset_output!(nl::NeighborList) = empty!(nl)

function CellListMap.reduce_output!(output::NeighborList, output_threaded::Vector{<:NeighborList})
    n_start = output.n
    n_tot = n_start
    for nb in output_threaded
        n_tot += nb.n
    end
    if length(output.list) < n_tot
        resize!(output.list, n_tot)
    end

    if (n_tot - n_start) > 100_000 && length(output_threaded) > 1 && Threads.nthreads() > 1
        Threads.@threads for i in eachindex(output_threaded)
            chunk_offset = n_start
            @inbounds for jb in 1:(i - 1)
                chunk_offset += output_threaded[jb].n
            end
            nb = output_threaded[i]
            if nb.n > 0
                copyto!(output.list, chunk_offset + 1, nb.list, 1, nb.n)
            end
        end
    else
        offset = n_start
        for nb in output_threaded
            if nb.n > 0
                copyto!(output.list, offset + 1, nb.list, 1, nb.n)
                offset += nb.n
            end
        end
    end

    output.n = n_tot
    return output
end

function find_neighbors(sys::System{D, AT},
                        nf::CellListMapNeighborFinder,
                        current_neighbors=sys.neighbor_finder.clm_particlesystem.neighbors, 
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where {D, AT}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    # Update the CellListMap.ParticleSystem
    positions = from_device(sys.coords)
    unitcell = first(clm_unitcell_arg(sys.boundary))
    parallel = (n_threads > 1)
    if isnothing(unitcell) # Avoid small Union dispatch
        CellListMap.update!(nf.clm_particlesystem; positions=positions, unitcell=nothing,
                            parallel=parallel)
    else
        CellListMap.update!(nf.clm_particlesystem; positions=positions, unitcell=unitcell,
                            parallel=parallel)
    end

    # Update the neighbor list
    eligible, special = kernel_matrix(nf.eligible), kernel_matrix(nf.special)
    neighbors = CellListMap.pairwise!(
        (p, neighbors) -> push_pair!(p, neighbors, eligible, special),
        nf.clm_particlesystem,
    )

    if AT <: AbstractGPUArray
        return NeighborList(neighbors.n, to_device(neighbors.list, AT))
    else
        return neighbors
    end
end

# Dense eligible and special matrices of a neighbor finder, taking memory proportional to
#   n_atoms^2, as used by the alchemical partitioning
function neighbor_finder_masks(nf::Union{GPUNeighborFinder, DistanceNeighborFinder,
                                         TreeNeighborFinder, CellListMapNeighborFinder})
    return copy_to_bitmatrix(from_device(nf.eligible)), copy_to_bitmatrix(from_device(nf.special))
end

function Base.show(io::IO, neighbor_finder::Union{DistanceNeighborFinder,
                                TreeNeighborFinder, CellListMapNeighborFinder})
    println(io, typeof(neighbor_finder))
    println(io, "  Size of eligible matrix = " , size(neighbor_finder.eligible))
    println(io, "  n_steps = " , neighbor_finder.n_steps)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end

function Base.show(io::IO, neighbor_finder::GPUNeighborFinder)
    println(io, typeof(neighbor_finder))
    println(io, "  n_atoms = " , neighbor_finder.n_atoms)
    println(io, "  n_excluded = " , length(neighbor_finder.excluded_i))
    println(io, "  n_special = " , length(neighbor_finder.special_i))
    println(io, "  n_steps_reorder = " , neighbor_finder.n_steps_reorder)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end
