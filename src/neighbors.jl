# Neighbor finders

export
    use_neighbors,
    NoNeighborFinder,
    find_neighbors,
    GPUNeighborFinder,
    GPUCellListNeighborFinder,
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
    if !isnothing(eligible) && !issymmetric(eligible)
        throw(ArgumentError("eligible matrix is not symmetric"))
    end
    if !isnothing(special) && !issymmetric(special)
        throw(ArgumentError("special matrix is not symmetric"))
    end
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
    GPUNeighborFinder(; n_atoms, dist_cutoff,
                      excluded_pairs=(), special_pairs=(), n_steps_reorder=10,
                      initialized=false, device_vector_type)
    GPUNeighborFinder(; eligible, dist_cutoff,
                      special=nothing, n_steps_reorder=10,
                      initialized=false, device_vector_type=nothing)

Neighbor finder for CUDA systems that uses Molly's tiled pairwise kernels.

`GPUNeighborFinder` does not materialize a conventional per-atom neighbor list.
Instead, the CUDA pairwise force and energy paths reorder atoms on a Morton
curve, convert sparse exclusions and special pairs into per-tile bitmasks, and
build a compact list of interacting 32x32 tiles directly on the device. For
that reason, [`find_neighbors`](@ref) returns `nothing` for this neighbor
finder.

This is the recommended neighbor finder for `CuArray` systems.

# Keyword arguments
- `n_atoms`: number of atoms when constructing directly from sparse exception
  pairs.
- `eligible`, `special`: compatibility inputs for dense boolean masks. These
  are converted once at construction into sparse exception lists; the dense
  masks are not retained.
- `dist_cutoff`: the neighbor search distance used by the pairwise kernels.
  This should be the interaction cutoff distance plus a buffer distance, since
  the tile list is only refreshed every `n_steps_reorder` steps. The buffer
  should be larger than the distance an atom can move in that time.
- `excluded_pairs`: iterable of `(i, j)` pairs that should be excluded from the
  normal nonbonded interaction path.
- `special_pairs`: iterable of `(i, j)` pairs that should use the "special"
  interaction path.
- `n_steps_reorder`: number of simulation steps between Morton reorder and
  tile-list refresh passes.
- `initialized`: whether the current sparse-mask preprocessing state can be
  reused.
- `device_vector_type`: concrete `AbstractVector{Int32}` storage type for
  sparse exception indices. It is required when constructing from `n_atoms`,
  and inferred from `eligible` when dense masks are provided.

# Notes
- Sparse exceptions are stored as four device vectors:
  `excluded_i`, `excluded_j`, `special_i`, and `special_j`.
- Updating the exception pairs should reset `initialized` so the tile masks are
  rebuilt on the next GPU force or energy evaluation.
"""
mutable struct GPUNeighborFinder{B, D, D2, E}
    n_atoms::B
    dist_cutoff::D
    dist_cutoff_2::D2
    n_steps_reorder::Int
    initialized::Bool
    cache_generation::UInt64
    excluded_i::E
    excluded_j::E
    special_i::E
    special_j::E
end

#=
    copy_to_bitmatrix(x)

Convert a given matrix `x` to a `BitMatrix`, copying if it is already one.

This is an internal utility used to ensure dense masks are efficiently handled
on the CPU before converting to sparse exceptions.
=#
copy_to_bitmatrix(x::BitMatrix) = copy(x)
copy_to_bitmatrix(x) = BitMatrix(Array(x))

#=
    to_bitmatrix(x)

Convert a given matrix `x` to a `BitMatrix`, returning it unchanged if it is already one.

Unlike `copy_to_bitmatrix` this does not copy, so the result should not be modified.
=#
to_bitmatrix(x::BitMatrix) = x
to_bitmatrix(x) = BitMatrix(Array(x))

#=
    gpu_exception_vector_type(eligible, device_vector_type)

Determine or validate the 1D `Int32` array type for storing sparse GPU exceptions.

This function ensures that the selected `device_vector_type` is compatible with the
provided `eligible` matrix (if on the GPU) or explicitly supplied.

# Arguments
- `eligible`: the dense boolean mask of eligible interactions.
- `device_vector_type`: an explicitly requested vector type or `nothing` to infer it.
=#
function gpu_exception_vector_type(eligible, device_vector_type)
    if eligible isa AbstractGPUArray
        return typeof(eligible).name.wrapper{Int32, 1}
    end
    return validate_device_vector_type(device_vector_type;
                                       missing_message="eligible must be on the GPU or device_vector_type must be provided")
end

function validate_device_vector_type(device_vector_type; missing_message::AbstractString="device_vector_type must be provided")
    if isnothing(device_vector_type)
        throw(ArgumentError(missing_message))
    end
    if !(device_vector_type isa Type && device_vector_type <: AbstractVector{Int32})
        throw(ArgumentError("device_vector_type must be a 1D Int32 array type, got $device_vector_type"))
    end
    return device_vector_type
end

#=
    normalize_pairs(pairs; allow_diagonal=false)

Normalize an iterable of pairs into a sorted, unique list of `(Int32, Int32)` tuples.

Each pair `(i, j)` is sorted such that `i < j` (unless `i == j`). If `allow_diagonal`
is `false`, any pairs where `i == j` are excluded. The resulting sequence contains
no duplicate pairs.

# Arguments
- `pairs`: an iterable of tuple pairs or 2-element arrays representing atom index pairs.
- `allow_diagonal::Bool=false`: whether to include `(i, i)` self-interactions.
=#
function normalize_pairs(pairs; allow_diagonal::Bool=false, n_atoms=nothing)
    normalized = Tuple{Int32, Int32}[]
    seen = Set{Tuple{Int32, Int32}}()
    n_atoms_32 = isnothing(n_atoms) ? nothing : Int32(n_atoms)
    for (i, j) in pairs
        i32 = Int32(i)
        j32 = Int32(j)
        if !isnothing(n_atoms_32) && !(Int32(1) <= i32 <= n_atoms_32 && Int32(1) <= j32 <= n_atoms_32)
            throw(ArgumentError("pair ($(Int(i32)), $(Int(j32))) is out of bounds for $n_atoms atoms"))
        end
        if j32 < i32
            i32, j32 = j32, i32
        end
        if i32 == j32 && !allow_diagonal
            continue
        end
        pair = (i32, j32)
        if pair ∉ seen
            push!(seen, pair)
            push!(normalized, pair)
        end
    end
    sort!(normalized)
    return normalized
end

#= 
    pair_list_vectors(pairs, ET)

Convert an iterable of pairs into two separate GPU device vectors of type `ET`.

Extracts the first and second elements of each pair into distinct arrays
and transfers them to the device using `Molly.to_device`.

# Arguments
- `pairs`: an iterable of `(i, j)` tuple pairs.
- `ET`: the target 1D `Int32` device vector type.
=#
function pair_list_vectors(pairs, ET)
    is = Int32[first(pair) for pair in pairs]
    js = Int32[last(pair) for pair in pairs]
    return Molly.to_device(is, ET), Molly.to_device(js, ET)
end

#=
    dense_masks_to_pair_lists(eligible_cpu, special_cpu)

Convert dense boolean matrices into sparse lists of `(i, j)` exclusion and special pairs.

This function identifies the entries where `eligible_cpu` is `false` to form excluded
pairs, and where `special_cpu` is `true` to form special interaction pairs. It only
retains the upper triangle indices (`i < j`).

# Arguments
- `eligible_cpu`: a dense boolean mask matrix indicating allowed standard interactions.
- `special_cpu`: a dense boolean mask matrix indicating special interactions.
=#
function dense_masks_to_pair_lists(eligible_cpu, special_cpu)
    all_exc = findall(.!eligible_cpu)
    excluded_pairs = Tuple{Int32, Int32}[]
    for idx in all_exc
        if idx[1] < idx[2]
            push!(excluded_pairs, (Int32(idx[1]), Int32(idx[2])))
        end
    end

    all_spec = findall(special_cpu)
    special_pairs = Tuple{Int32, Int32}[]
    for idx in all_spec
        if idx[1] < idx[2]
            push!(special_pairs, (Int32(idx[1]), Int32(idx[2])))
        end
    end
    return excluded_pairs, special_pairs
end

function neighbor_finder_masks(nf::GPUNeighborFinder)
    eligible = trues(nf.n_atoms, nf.n_atoms)
    special = falses(nf.n_atoms, nf.n_atoms)
    for i in 1:nf.n_atoms
        eligible[i, i] = false
    end
    for (i, j) in zip(from_device(nf.excluded_i), from_device(nf.excluded_j))
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, j) in zip(from_device(nf.special_i), from_device(nf.special_j))
        special[i, j] = true
        special[j, i] = true
    end
    return eligible, special
end

neighbor_finder_masks(nf, ::Integer) = neighbor_finder_masks(nf)

function neighbor_finder_masks(::NoNeighborFinder, n_atoms::Integer)
    eligible = trues(n_atoms, n_atoms)
    special = falses(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    return eligible, special
end

#=
    update_sparse_pairs!(nf, excluded_pairs, special_pairs)

Replace the existing exception lists in a [`GPUNeighborFinder`](@ref) with new ones.

This normalizes the `excluded_pairs` and `special_pairs`, uploads them to the
device, and resets the neighbor finder's `initialized` flag so that internal
GPU interaction masks are rebuilt.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `excluded_pairs`: an iterable of `(i, j)` pairs that should be excluded.
- `special_pairs`: an iterable of `(i, j)` pairs that should use the special interaction path.
=#
function update_sparse_pairs!(nf::GPUNeighborFinder, excluded_pairs, special_pairs)
    ET = typeof(nf.excluded_i)
    excluded_pairs = normalize_pairs(excluded_pairs; n_atoms=nf.n_atoms)
    special_pairs = normalize_pairs(special_pairs; n_atoms=nf.n_atoms)
    nf.excluded_i, nf.excluded_j = pair_list_vectors(excluded_pairs, ET)
    nf.special_i, nf.special_j = pair_list_vectors(special_pairs, ET)
    nf.initialized = false
    nf.cache_generation += 0x0000000000000001
    return nf
end

#=
    append_excluded_pairs!(nf, pairs)

Append new excluded pairs to the existing exclusions in a [`GPUNeighborFinder`](@ref).

This downloads the current sparse lists from the device, concatenates them with
the new `pairs`, and then calls `update_sparse_pairs!` to normalize and upload
everything back to the GPU. The `initialized` state will be reset.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `pairs`: an iterable of `(i, j)` pairs to add to the exclusions list.
=#
function append_excluded_pairs!(nf::GPUNeighborFinder, pairs)
    existing_pairs = collect(zip(from_device(nf.excluded_i), from_device(nf.excluded_j)))
    update_sparse_pairs!(nf, vcat(existing_pairs, collect(pairs)),
                         collect(zip(from_device(nf.special_i), from_device(nf.special_j))))
    return nf
end

function GPUNeighborFinder(;
                            n_atoms=nothing,
                            eligible=nothing,
                            dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            special=nothing,
                            n_steps_reorder=10,
                            initialized=false,
                            device_vector_type=nothing)
    if !isnothing(n_atoms)
        ET = validate_device_vector_type(device_vector_type)
        excluded_pairs_norm = normalize_pairs(excluded_pairs; n_atoms=n_atoms)
        special_pairs_norm = normalize_pairs(special_pairs; n_atoms=n_atoms)
        excluded_i, excluded_j = pair_list_vectors(excluded_pairs_norm, ET)
        special_i, special_j = pair_list_vectors(special_pairs_norm, ET)
        dist_cutoff_2 = dist_cutoff^2
        return GPUNeighborFinder{Int, typeof(dist_cutoff), typeof(dist_cutoff_2), typeof(excluded_i)}(
                    Int(n_atoms), dist_cutoff, dist_cutoff_2, n_steps_reorder, initialized, 0,
                    excluded_i, excluded_j, special_i, special_j)
    end

    check_neighbor_matrices(eligible, special)
    isnothing(eligible) && throw(ArgumentError("either n_atoms or eligible must be provided"))
    ET = gpu_exception_vector_type(eligible, device_vector_type)
    if isnothing(special)
        special = zero(eligible)
    end
    eligible_cpu = copy_to_bitmatrix(eligible)
    special_cpu = copy_to_bitmatrix(special)
    if !(size(eligible_cpu) == size(special_cpu))
        throw(ArgumentError("eligible and special must have the same size"))
    end
    excluded_pairs_cpu, special_pairs_cpu = dense_masks_to_pair_lists(eligible_cpu, special_cpu)
    return GPUNeighborFinder(
        n_atoms=size(eligible_cpu, 1),
        eligible=nothing,
        dist_cutoff=dist_cutoff,
        excluded_pairs=excluded_pairs_cpu,
        special_pairs=special_pairs_cpu,
        n_steps_reorder=n_steps_reorder,
        initialized=initialized,
        device_vector_type=ET,
    )
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
    GPUCellListNeighborFinder(;
        dist_cutoff,
        n_steps=10,
        max_neighbors=640,
        output=:ragged,
    )

GPU cell-list neighbor finder that materializes a per-atom geometric
neighbor list on the GPU.

`output=:ragged` returns only the per-atom padded ragged representation.
`output=:ragged_and_pairs` additionally constructs a flat geometric
half-pair list. Geometric pairs do not include force-field exclusions
or special-pair flags.
"""
struct GPUCellListNeighborFinder{D}
    dist_cutoff::D
    n_steps::Int
    max_neighbors::Int
    output::Symbol
end

function GPUCellListNeighborFinder(;
    dist_cutoff,
    n_steps=10,
    max_neighbors=640,
    output::Symbol=:ragged,
)
    output in (:ragged, :ragged_and_pairs) || throw(
        ArgumentError(
            "output must be :ragged or :ragged_and_pairs, got $output",
        ),
    )

    return GPUCellListNeighborFinder(
        dist_cutoff,
        Int(n_steps),
        Int(max_neighbors),
        output,
    )
end

"""
    DistanceNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance.

This is the recommended neighbor finder on non-NVIDIA GPUs.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
"""
struct DistanceNeighborFinder{B, D}
    eligible::B
    dist_cutoff::D
    special::B
    n_steps::Int
end

function DistanceNeighborFinder(;
                                eligible,
                                dist_cutoff,
                                special=zero(eligible),
                                n_steps=10)
    check_neighbor_matrices(eligible, special)
    return DistanceNeighborFinder{typeof(eligible), typeof(dist_cutoff)}(
                eligible, dist_cutoff, special, n_steps)
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
                                        special_i, list_offset)
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
                            (@view nf.eligible[:, i]), i - 1, Val(D))
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
                                   (@view nf.special[:, i]), list_starts[i])
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

@kernel inbounds=true function distance_neighbor_finder_mask_kernel!(masks, counts, @Const(coords),
                                        @Const(eligible), boundary, sq_dist_cutoff, n_inters)
    n_atoms = length(coords)
    word_i = @index(Global, Linear)

    if word_i <= length(masks)
        mask = zero(UInt32)
        for bit_i in 0:(n_pairs_per_mask_gpu - 1)
            inter_i = mask_bit_pair_index(word_i, bit_i)
            if inter_i <= n_inters
                i, j = pair_index_col(n_atoms, inter_i)
                if eligible[i, j]
                    dr = vector(coords[i], coords[j], boundary)
                    r2 = sum(abs2, dr)
                    if r2 <= sq_dist_cutoff
                        mask |= (one(UInt32) << bit_i)
                    end
                end
            end
        end
        masks[word_i] = mask
        counts[word_i] = Int32(count_ones(mask))
    end
end

@kernel inbounds=true function distance_neighbor_finder_fill_kernel!(neighbors_list, @Const(masks),
                                        @Const(list_ends), @Const(special), n_atoms)
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
                neighbors_list[ni] = (Int32(j), Int32(i), special[j, i])
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
    mask_kernel!(masks, counts, sys.coords, nf.eligible, sys.boundary, nf.dist_cutoff^2,
                 n_inters; ndrange=n_masks)

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
        fill_kernel!(neighbors_list, masks, counts, nf.special, length(sys); ndrange=n_masks)
    end

    return NeighborList(n_neighbors, neighbors_list)
end

"""
    TreeNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance using a tree search.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.

Can not be used if one or more dimensions has infinite boundaries.
Can not be used with [`TriclinicBoundary`](@ref).
"""
struct TreeNeighborFinder{D}
    eligible::BitArray{2}
    dist_cutoff::D
    special::BitArray{2}
    n_steps::Int
end

function TreeNeighborFinder(;
                            eligible,
                            dist_cutoff,
                            special=zero(eligible),
                            n_steps=10)
    check_neighbor_matrices(eligible, special)
    return TreeNeighborFinder(eligible, dist_cutoff, special, n_steps)
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

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(sys)
            ci = ustrip.(sys.coords[i])
            nbi = @view nf.eligible[:, i]
            speci = @view nf.special[:, i]
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(nl_threads[chunk_i], (Int32(i), Int32(j), speci[j]))
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
    CellListMapNeighborFinder(; eligible, dist_cutoff, boundary,
                                special, n_steps, x0, number_of_batches)

Find close atoms by distance using a cell list algorithm from CellListMap.jl.

This is the recommended neighbor finder on CPU.
`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
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
mutable struct CellListMapNeighborFinder{N, T, S}
    eligible::BitArray{2}
    dist_cutoff::T
    special::BitArray{2}
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
                                   eligible,
                                   dist_cutoff::T,
                                   boundary::AbstractBoundary,
                                   special=zero(eligible),
                                   n_steps=10,
                                   x0=nothing,
                                   number_of_batches=(0, 0)) where T
    check_neighbor_matrices(eligible, special)
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

    return CellListMapNeighborFinder{D, T, typeof(clm_system)}(eligible, dist_cutoff, special, n_steps, clm_system)
end

# Add a pair to the pair list
# If the buffer size is large enough update the element, otherwise push a new element
#   to `neighbors.list`
@inline function push_pair!(pair, neighbors::NeighborList, eligible, special)
    i, j = pair.i, pair.j
    @inbounds if eligible[i, j]
        n = neighbors.n + 1
        neighbors.n = n
        list = neighbors.list
        element = (i, j, special[i, j])
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
CellListMap.reducer(nl1::NeighborList, nl2::NeighborList) = append!(nl1, nl2)

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
    neighbors = CellListMap.pairwise!(
        (p, neighbors) -> push_pair!(p, neighbors, nf.eligible, nf.special), 
        nf.clm_particlesystem,
    )

    if AT <: AbstractGPUArray
        return NeighborList(neighbors.n, to_device(neighbors.list, AT))
    else
        return neighbors
    end
end

function neighbor_finder_masks(nf::Union{DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder})
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
