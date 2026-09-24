# O(3) irreducible-representation bookkeeping for equivariant potentials (e.g. Allegro).
#
# An irrep is labelled by (l, p): l is the rotation order (0, 1, 2, ...) with dimension 2l+1,
# p is the parity (+1 "even"/e, -1 "odd"/o) under spatial inversion. A feature vector carries a
# direct sum of irreps, each with an integer multiplicity (number of channels). This mirrors the
# e3nn `o3.Irrep` / `o3.Irreps` types so that trained e3nn weights can be transferred.
#
# These types are pure bookkeeping (no Lux/HDF5), so they live in core Molly and can be used and
# tested with a bare `using Molly`.
#
# These names are internal (unexported); reference them as `Molly.Irreps`, `Molly.real_sph_harm`,
# etc. Only the user-facing `AllegroPotential` is exported (from ml_potentials.jl).

"""
    Irrep(l, p)

A single O(3) irreducible representation of rotation order `l` (`≥ 0`) and parity `p`
(`+1` even, `-1` odd). Its dimension is `2l + 1`. Under a rotation the `2l+1` components mix by
the Wigner-D matrix `D^l`; under inversion they pick up a factor `p`.
"""
struct Irrep
    l::Int
    p::Int
    function Irrep(l::Integer, p::Integer)
        l >= 0 || throw(ArgumentError("irrep order l must be ≥ 0, got $l"))
        (p == 1 || p == -1) || throw(ArgumentError("irrep parity p must be ±1, got $p"))
        new(Int(l), Int(p))
    end
end

"Dimension `2l+1` of an irrep."
irrep_dim(ir::Irrep) = 2 * ir.l + 1

Base.show(io::IO, ir::Irrep) = print(io, ir.l, ir.p == 1 ? "e" : "o")

"""
    MulIrrep(mul, ir)

An irrep `ir` repeated with multiplicity `mul` (the number of channels carrying it).
"""
struct MulIrrep
    mul::Int
    ir::Irrep
end

irrep_dim(mi::MulIrrep) = mi.mul * irrep_dim(mi.ir)

Base.show(io::IO, mi::MulIrrep) = (print(io, mi.mul, "x"); show(io, mi.ir))

"""
    Irreps(entries)
    Irreps("32x0e + 32x1o + 16x2e")

An ordered list of `MulIrrep` entries describing the layout of an equivariant feature vector.

The flat feature vector concatenates the entries in order. Within an entry of multiplicity `mul`
and irrep dimension `d = 2l+1` the layout is **channel-major**: the `mul` channels are the outer
index and the `2l+1` components the inner index (matching e3nn's convention, so a weight tensor
exported from e3nn maps in without a reshape). The element for channel `c` (`1:mul`) and component
`m` (`1:d`) of entry `k` sits at `offset[k] + (c-1)*d + m` in the flat vector.

The order is preserved exactly and never sorted: it must match the order the trained model presents
its irreps in, otherwise loaded weights are scrambled.
"""
struct Irreps
    entries::Vector{MulIrrep}
    offsets::Vector{Int}   # offsets[k] = start-1 of entry k in the flat vector (0-based prefix)
    dim::Int               # total dimension Σ mul*(2l+1)
end

function Irreps(entries::AbstractVector{MulIrrep})
    offsets = Vector{Int}(undef, length(entries))
    off = 0
    for (k, mi) in enumerate(entries)
        offsets[k] = off
        off += irrep_dim(mi)
    end
    return Irreps(collect(entries), offsets, off)
end

# Parse an e3nn-style string like "32x0e+16x1o + 8x2e".
function Irreps(s::AbstractString)
    entries = MulIrrep[]
    for tok in split(s, '+')
        t = strip(tok)
        isempty(t) && continue
        mul_str, ir_str = occursin('x', t) ? split(t, 'x'; limit=2) : ("1", t)
        mul = parse(Int, strip(mul_str))
        ir_str = strip(ir_str)
        pc = ir_str[end]
        p = pc == 'e' ? 1 : pc == 'o' ? -1 : throw(ArgumentError("bad parity in irrep '$ir_str'"))
        l = parse(Int, ir_str[1:end-1])
        push!(entries, MulIrrep(mul, Irrep(l, p)))
    end
    return Irreps(entries)
end

Base.length(irs::Irreps) = length(irs.entries)
Base.getindex(irs::Irreps, k::Integer) = irs.entries[k]
Base.iterate(irs::Irreps, st=1) = st > length(irs) ? nothing : (irs.entries[st], st + 1)
"Total dimension of the flat feature vector."
dim(irs::Irreps) = irs.dim

function Base.show(io::IO, irs::Irreps)
    print(io, "Irreps(\"")
    for (k, mi) in enumerate(irs.entries)
        k > 1 && print(io, "+")
        show(io, mi)
    end
    print(io, "\")")
end

"""
    entry_range(irs, k)

The `UnitRange` of flat-vector indices spanned by entry `k` of `irs`.
"""
entry_range(irs::Irreps, k::Integer) = (irs.offsets[k] + 1):(irs.offsets[k] + irrep_dim(irs.entries[k]))

"""
    component_index(irs, k, c, m)

Flat-vector index of channel `c` (`1:mul`), component `m` (`1:2l+1`) of entry `k`, using the
channel-major layout described in [`Irreps`](@ref).
"""
@inline function component_index(irs::Irreps, k::Integer, c::Integer, m::Integer)
    d = irrep_dim(irs.entries[k].ir)
    return irs.offsets[k] + (c - 1) * d + m
end

"""
    TensorProductPaths(irreps_in1, irreps_in2, irreps_out; mode=:uvu)

Enumerate the allowed coupling paths for a Clebsch-Gordan tensor product
`irreps_in1 ⊗ irreps_in2 → irreps_out`. A path connects entry `k1` of `in1` and entry `k2` of
`in2` to entry `k3` of `out` when the angular-momentum selection rule
`|l1 − l2| ≤ l3 ≤ l1 + l2` and the parity rule `p1 * p2 == p3` both hold.

Connection `mode`:
- `:uvu` (Allegro's default, "channel-wise"): `in1` carries the equivariant feature with
  `n_channels`, `in2` (typically the spherical harmonics) has multiplicity 1 per irrep, and the
  output keeps `in1`'s multiplicity. Each path owns one weight **per channel**.

`weight_offset[p]` gives the start (0-based) of path `p`'s weights in the flat per-edge weight
vector; `n_weights` is the total. `l` stores `(l1, l2, l3)` per path; `k` stores the entry indices
`(k1, k2, k3)`.
"""
struct TensorProductPaths
    in1::Irreps
    in2::Irreps
    out::Irreps
    k::Vector{NTuple{3,Int}}        # (k1, k2, k3) entry indices
    l::Vector{NTuple{3,Int}}        # (l1, l2, l3)
    weight_offset::Vector{Int}      # 0-based offset of this path's weights in the flat weight vec
    n_weights_per_path::Vector{Int} # channels per path (uvu ⇒ mul of in1 entry)
    n_weights::Int
    mode::Symbol
end

function TensorProductPaths(in1::Irreps, in2::Irreps, out::Irreps; mode::Symbol=:uvu)
    mode == :uvu || throw(ArgumentError("only :uvu connection mode is supported, got $mode"))
    ks = NTuple{3,Int}[]
    ls = NTuple{3,Int}[]
    woff = Int[]
    npw = Int[]
    off = 0
    for (k1, mi1) in enumerate(in1.entries), (k2, mi2) in enumerate(in2.entries)
        l1, p1 = mi1.ir.l, mi1.ir.p
        l2, p2 = mi2.ir.l, mi2.ir.p
        for k3 in eachindex(out.entries)
            l3, p3 = out.entries[k3].ir.l, out.entries[k3].ir.p
            (abs(l1 - l2) <= l3 <= l1 + l2) || continue
            (p1 * p2 == p3) || continue
            # uvu: in1 mul must equal out mul on this path
            mi1.mul == out.entries[k3].mul ||
                throw(ArgumentError("uvu path $k1⊗$k2→$k3 needs matching multiplicities " *
                                    "($(mi1.mul) vs $(out.entries[k3].mul))"))
            push!(ks, (k1, k2, k3))
            push!(ls, (l1, l2, l3))
            push!(woff, off)
            push!(npw, mi1.mul)
            off += mi1.mul
        end
    end
    return TensorProductPaths(in1, in2, out, ks, ls, woff, npw, off, mode)
end

Base.length(tp::TensorProductPaths) = length(tp.k)
