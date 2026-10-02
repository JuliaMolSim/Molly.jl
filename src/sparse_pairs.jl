# Sparse storage of atom pairs, used for the eligible and special matrices of neighbor finders

export SparsePairMatrix

"""
    SparsePairMatrix(n_atoms, pairs; listed=true, array_type=Array)

A symmetric `n_atoms` x `n_atoms` `AbstractMatrix{Bool}` that stores a list of atom pairs.

Listed pairs have the value `listed`, all other off-diagonal entries have the value
`!listed` and the diagonal is `false`.
This makes it a sparse alternative to the dense `eligible` and `special` matrices given
to neighbor finders, which take memory proportional to `n_atoms^2`:
- `SparsePairMatrix(n_atoms, excluded_pairs; listed=false)` is an eligible matrix where
  every pair of different atoms is eligible apart from the excluded pairs.
- `SparsePairMatrix(n_atoms, special_pairs; listed=true)` is a special matrix where only
  the given pairs are special.
Neighbor finders build these matrices when they are given `n_atoms`, `excluded_pairs`
and `special_pairs`, so usually they do not have to be constructed directly.

`pairs` is an iterable of `(i, j)` pairs, in either order and with duplicates allowed, or
a sparse matrix whose stored `true` entries are the pairs.
`array_type` is the array type used to store the lists, for example `CuArray` to use the
matrix with a neighbor finder on the GPU.

For each atom the atoms it is paired with are stored in ascending order, so the memory
used is proportional to the number of pairs and looking up an entry means scanning the
partners of one atom.
Indexing into a matrix stored on the GPU from the CPU is not allowed, as for other GPU
arrays.
"""
mutable struct SparsePairMatrix{V <: AbstractVector{Int32}} <: AbstractMatrix{Bool}
    n_atoms::Int
    listed::Bool
    # The partners of atom i are partners[starts[i]:(starts[i + 1] - 1)], in ascending order
    # Each pair is stored under both of its atoms
    starts::V
    partners::V
end

function SparsePairMatrix(n_atoms::Integer, pairs; listed::Bool=true, array_type=Array)
    starts, partners = pair_lists_csr(normalize_pairs(pairs; n_atoms=n_atoms), n_atoms)
    return SparsePairMatrix(Int(n_atoms), listed, to_device(starts, array_type),
                            to_device(partners, array_type))
end

Base.size(m::SparsePairMatrix) = (m.n_atoms, m.n_atoms)
Base.IndexStyle(::Type{<:SparsePairMatrix}) = IndexCartesian()

Base.@propagate_inbounds function Base.getindex(m::SparsePairMatrix, i::Int, j::Int)
    @boundscheck checkbounds(m, i, j)
    return (i != j) && (pair_listed(m.starts, m.partners, i, j) == m.listed)
end

LinearAlgebra.issymmetric(::SparsePairMatrix) = true

Base.copy(m::SparsePairMatrix) = SparsePairMatrix(m.n_atoms, m.listed, copy(m.starts),
                                                  copy(m.partners))

# All entries false, i.e. a special matrix with no special pairs, stored like `m`
function Base.zero(m::SparsePairMatrix)
    starts = similar(m.starts, m.n_atoms + 1)
    fill!(starts, one(Int32))
    return SparsePairMatrix(m.n_atoms, true, starts, similar(m.partners, 0))
end

function Base.show(io::IO, ::MIME"text/plain", m::SparsePairMatrix)
    print(io, m.n_atoms, "×", m.n_atoms, " SparsePairMatrix{", typeof(m.starts), "} with ",
          n_listed_pairs(m), " pairs listed as ", m.listed)
end

from_device(m::SparsePairMatrix) = SparsePairMatrix(m.n_atoms, m.listed,
                                            from_device(m.starts), from_device(m.partners))

function to_device(m::SparsePairMatrix, ::Type{AT}) where AT
    return SparsePairMatrix(m.n_atoms, m.listed, to_device(m.starts, AT),
                            to_device(m.partners, AT))
end

# Whether the matrix is stored on the GPU, used when checking neighbor finders
neighbor_matrix_on_gpu(m) = m isa AbstractGPUArray
neighbor_matrix_on_gpu(m::SparsePairMatrix) = m.starts isa AbstractGPUArray

# The number of pairs listed in a [`SparsePairMatrix`](@ref), counting each pair once.
n_listed_pairs(m::SparsePairMatrix) = length(m.partners) ÷ 2

#=
The number of pairs `(i, j)`, `i < j`, that are true in a symmetric eligible or special
matrix, counting each pair once, i.e. the number of pairs returned by `true_pairs`.
=#
function n_true_pairs(m::SparsePairMatrix)
    return m.listed ? n_listed_pairs(m) : n_atoms_to_n_pairs(m.n_atoms) - n_listed_pairs(m)
end

function n_true_pairs(m::AbstractMatrix)
    # The matrix is symmetric, so each off-diagonal pair is counted twice by the sum, and
    #   the diagonal is not a pair of different atoms so is not counted at all
    return (sum(m) - sum(@view m[diagind(m)])) ÷ 2
end

#=
Whether atom `i` is in the sorted partner list of atom `j`.
Works on CPU and inside GPU kernels. The lists are short, a few entries for most atoms,
so a linear scan that stops at the first partner not below `i` is used.
=#
@inline function pair_listed(starts, partners, i, j)
    @inbounds k_start = starts[j]
    @inbounds k_end = starts[j + 1]
    k = k_start
    while k < k_end
        @inbounds p = partners[k]
        if p >= i
            return p == i
        end
        k += one(k)
    end
    return false
end

#=
The form of an eligible or special matrix passed to GPU kernels.
A dense matrix is passed as it is. A SparsePairMatrix is passed as a NamedTuple of its
arrays, which the kernel launch converts to device arrays, and read with `pair_value`.
=#
kernel_matrix(m::AbstractMatrix) = m
kernel_matrix(m::SparsePairMatrix) = (starts=m.starts, partners=m.partners, listed=m.listed)

const SparsePairKernelMatrix = NamedTuple{(:starts, :partners, :listed)}

# The value of entry (i, j), i != j, of an eligible or special matrix inside a kernel
@inline pair_value(m::AbstractMatrix, i, j) = @inbounds m[i, j]
@inline function pair_value(m::SparsePairKernelMatrix, i, j)
    return pair_listed(m.starts, m.partners, i, j) == m.listed
end

#=
Build per-atom partner lists from pairs normalized with `normalize_pairs`, i.e. sorted
`(i, j)` pairs with `i < j` and no duplicates.
Since the pairs are sorted, filling the lists in pair order leaves each list sorted: the
partners `i < a` of atom `a` come from pairs before the pairs `(a, j)` with `j > a`.
=#
function pair_lists_csr(pairs_norm, n_atoms::Integer)
    starts = zeros(Int32, n_atoms + 1)
    for (i, j) in pairs_norm
        starts[i + 1] += Int32(1)
        starts[j + 1] += Int32(1)
    end
    starts[1] = Int32(1)
    for a in 1:n_atoms
        starts[a + 1] += starts[a]
    end
    partners = Vector{Int32}(undef, 2 * length(pairs_norm))
    next = starts[1:n_atoms]
    for (i, j) in pairs_norm
        partners[next[i]] = j
        next[i] += Int32(1)
        partners[next[j]] = i
        next[j] += Int32(1)
    end
    return starts, partners
end

# The listed pairs of a [`SparsePairMatrix`](@ref) as a sorted `Vector{Tuple{Int32, Int32}}`
#   of `(i, j)` pairs with `i < j`.
function listed_pairs(m::SparsePairMatrix)
    starts, partners = from_device(m.starts), from_device(m.partners)
    pairs = Vector{Tuple{Int32, Int32}}(undef, length(partners) ÷ 2)
    n_found = 0
    for i in 1:m.n_atoms
        for k in starts[i]:(starts[i + 1] - Int32(1))
            j = partners[k]
            if j > i
                n_found += 1
                pairs[n_found] = (Int32(i), j)
            end
        end
    end
    return pairs
end

# add_listed_pairs!(m::SparsePairMatrix, pairs)
# Add pairs to the list of a [`SparsePairMatrix`](@ref), in place.
# For a matrix with `listed=false`, such as an eligible matrix built from excluded pairs,
#   this excludes the pairs.
function add_listed_pairs!(m::SparsePairMatrix, pairs)
    AT = typeof(m.starts)
    all_pairs = vcat(listed_pairs(m), collect_pairs(pairs))
    starts, partners = pair_lists_csr(normalize_pairs(all_pairs; n_atoms=m.n_atoms),
                                      m.n_atoms)
    m.starts, m.partners = to_device(starts, AT), to_device(partners, AT)
    return m
end

# Remove pairs from the list of a [`SparsePairMatrix`](@ref), in place.
function remove_listed_pairs!(m::SparsePairMatrix, pairs)
    AT = typeof(m.starts)
    to_remove = Set(normalize_pairs(pairs; n_atoms=m.n_atoms))
    kept = filter(p -> !(p in to_remove), listed_pairs(m))
    starts, partners = pair_lists_csr(kept, m.n_atoms)
    m.starts, m.partners = to_device(starts, AT), to_device(partners, AT)
    return m
end

# Make pairs ineligible in a sparse eligible matrix, in place.
function exclude_pairs!(eligible::SparsePairMatrix, pairs)
    if eligible.listed
        return remove_listed_pairs!(eligible, pairs)
    else
        return add_listed_pairs!(eligible, pairs)
    end
end

# Convert an iterable of pairs, or a sparse matrix whose true entries are the pairs, to a
#   vector of pairs
collect_pairs(pairs) = collect(Tuple{Int32, Int32}, (Int32(p[1]), Int32(p[2])) for p in pairs)
collect_pairs(pairs::Vector{Tuple{Int32, Int32}}) = pairs

function collect_pairs(pairs::AbstractSparseMatrix)
    is, js, vals = findnz(pairs)
    return [(Int32(i), Int32(j)) for (i, j, v) in zip(is, js, vals) if v]
end

#=
    normalize_pairs(pairs; allow_diagonal=false, n_atoms=nothing)

Normalize pairs into a sorted, unique list of `(Int32, Int32)` tuples with `i < j`.

Pairs are given as an iterable of tuples or 2-element collections, or as a sparse matrix
whose true entries are the pairs.
Pairs with `i == j` are dropped unless `allow_diagonal` is `true`.
If `n_atoms` is given, an `ArgumentError` is thrown for a pair out of bounds.
=#
function normalize_pairs(pairs; allow_diagonal::Bool=false, n_atoms=nothing)
    normalized = Tuple{Int32, Int32}[]
    n_atoms_32 = isnothing(n_atoms) ? nothing : Int32(n_atoms)
    for (i, j) in collect_pairs(pairs)
        if !isnothing(n_atoms_32) && !(Int32(1) <= i <= n_atoms_32 && Int32(1) <= j <= n_atoms_32)
            throw(ArgumentError("pair ($(Int(i)), $(Int(j))) is out of bounds for $n_atoms atoms"))
        end
        if j < i
            i, j = j, i
        end
        if i == j && !allow_diagonal
            continue
        end
        push!(normalized, (i, j))
    end
    sort!(normalized)
    unique!(normalized)
    return normalized
end

#=
Pairs `(i, j)`, `i < j`, of an eligible or special matrix that are not eligible or are
special respectively, as used to find the excluded pairs of the Ewald methods and to
convert dense masks to sparse ones.
For a SparsePairMatrix that lists those pairs they are read from the list, otherwise
every entry is scanned, which takes memory and time proportional to `n_atoms^2`.
=#
function ineligible_pairs(eligible::SparsePairMatrix)
    return eligible.listed ? upper_pairs_where(!, eligible) : listed_pairs(eligible)
end

function true_pairs(special::SparsePairMatrix)
    return special.listed ? listed_pairs(special) : upper_pairs_where(identity, special)
end

ineligible_pairs(eligible::AbstractMatrix) = upper_pairs_where(!, eligible)
true_pairs(special::AbstractMatrix) = upper_pairs_where(identity, special)

# Pairs `(i, j)`, `i < j`, where `f(matrix[i, j])` is true, scanning 64 entries at a time
function upper_pairs_where(f, matrix::AbstractMatrix)
    mat_cpu = to_bitmatrix(from_device(matrix))
    n_atoms = size(mat_cpu, 1)
    pairs = Tuple{Int32, Int32}[]
    n_entries = n_atoms * n_atoms
    iszero(n_entries) && return pairs
    chunks = mat_cpu.chunks
    n_chunks = length(chunks)
    # Bits past the end of the last chunk are unset in a BitArray but can be set by the
    #   negation, so mask them off
    end_mask = ~zero(UInt64) >>> ((-n_entries) & 63)
    for ci in 1:n_chunks
        chunk = (f === identity ? chunks[ci] : ~chunks[ci])
        if ci == n_chunks
            chunk &= end_mask
        end
        while !iszero(chunk)
            # Column-major linear index of the set bit, zero-based
            li = (ci - 1) * 64 + trailing_zeros(chunk)
            j, i = divrem(li, n_atoms)
            if i < j
                push!(pairs, (Int32(i + 1), Int32(j + 1)))
            end
            chunk &= chunk - one(UInt64)
        end
    end
    # The scan runs down the columns, sort to give the same order as looping over i then j
    sort!(pairs)
    return pairs
end

#=
Convert an eligible and a special matrix, dense or sparse, to sparse ones that list the
excluded and the special pairs respectively, as the CUDA tiled kernels read them.
A SparsePairMatrix that already lists those pairs is moved to `AT`, anything else is
converted by scanning it.
=#
function sparse_eligible(eligible::AbstractMatrix, n_atoms, AT)
    if eligible isa SparsePairMatrix && !eligible.listed
        return to_device(eligible, AT)
    end
    return SparsePairMatrix(n_atoms, ineligible_pairs(eligible); listed=false, array_type=AT)
end

function sparse_special(special::AbstractMatrix, n_atoms, AT)
    if special isa SparsePairMatrix && special.listed
        return to_device(special, AT)
    end
    return SparsePairMatrix(n_atoms, true_pairs(special); listed=true, array_type=AT)
end

sparse_special(::Nothing, n_atoms, AT) = SparsePairMatrix(n_atoms, (); listed=true,
                                                          array_type=AT)

# Convert to a BitMatrix, using the sparse structure where possible
copy_to_bitmatrix(x::BitMatrix) = copy(x)
copy_to_bitmatrix(x) = BitMatrix(Array(x))
copy_to_bitmatrix(m::SparsePairMatrix) = dense_bitmatrix(m)

to_bitmatrix(x::BitMatrix) = x
to_bitmatrix(x) = BitMatrix(Array(x))
to_bitmatrix(m::SparsePairMatrix) = dense_bitmatrix(m)

function dense_bitmatrix(m::SparsePairMatrix)
    dense = (m.listed ? falses(m.n_atoms, m.n_atoms) : trues(m.n_atoms, m.n_atoms))
    for i in 1:m.n_atoms
        dense[i, i] = false
    end
    for (i, j) in listed_pairs(m)
        dense[i, j] = m.listed
        dense[j, i] = m.listed
    end
    return dense
end
