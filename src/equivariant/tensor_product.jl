# Weighted Clebsch-Gordan tensor product (uvu / channel-wise) and its analytic VJP.
#
# Given a per-edge equivariant feature `x` (irreps `in1`, channel-major flat layout), a per-edge
# spherical-harmonic vector `y` (irreps `in2`, multiplicity 1), and per-edge, per-path, per-channel
# weights `w`, produce the output feature `z` (irreps `out`):
#
#   z[k3,c,m3] = Σ_{paths p→k3} w[p,c] · Σ_{(m1,m2,m3)∈p} CG · x[k1,c,m1] · y[k2,m2]
#
# The product is bilinear in (x, y) and linear in w, so all three adjoints reuse the same sparse
# CG triples. This is the CPU reference; the GPU (KernelAbstractions) version is a later milestone.
# Pure maths, no Lux/HDF5 — core Molly. Internal (unexported).

# flat index of (channel c, component m) within entry k of channel-major irreps
@inline _idx(irs::Irreps, k::Int, c::Int, m::Int) = irs.offsets[k] + (c - 1) * (2 * irs.entries[k].ir.l + 1) + m

"""
    tensor_product(paths, cg, x, y, w) -> z

Forward weighted CG tensor product. `x`, `y`, `w` are flat vectors (`x`: `dim(in1)`, `y`:
`dim(in2)`, `w`: `paths.n_weights`). Returns `z` of length `dim(out)`.
"""
function tensor_product(paths::TensorProductPaths, cg::SparseCG{T},
                        x::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T}) where T
    out = paths.out
    z = zeros(T, out.dim)
    in1, in2 = paths.in1, paths.in2
    for p in eachindex(paths.k)
        k1, k2, k3 = paths.k[p]
        nch = paths.n_weights_per_path[p]
        woff = paths.weight_offset[p]
        rng = (cg.poff[p] + 1):cg.poff[p + 1]
        for c in 1:nch
            wc = w[woff + c]
            wc == 0 && continue
            @inbounds for t in rng
                m1 = Int(cg.m1[t]); m2 = Int(cg.m2[t]); m3 = Int(cg.m3[t]); v = cg.val[t]
                z[_idx(out, k3, c, m3)] += wc * v * x[_idx(in1, k1, c, m1)] * y[_idx(in2, k2, 1, m2)]
            end
        end
    end
    return z
end

"""
    tensor_product_vjp(paths, cg, x, y, w, ḡ) -> (x̄, ȳ, w̄)

Reverse pass: given the output adjoint `ḡ = ∂E/∂z`, return the input adjoints `x̄ = ∂E/∂x`,
`ȳ = ∂E/∂y`, and weight adjoint `w̄ = ∂E/∂w`, all reusing the same CG triples.
"""
function tensor_product_vjp(paths::TensorProductPaths, cg::SparseCG{T},
                            x::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T},
                            ḡ::AbstractVector{T}) where T
    in1, in2, out = paths.in1, paths.in2, paths.out
    x̄ = zeros(T, in1.dim)
    ȳ = zeros(T, in2.dim)
    w̄ = zeros(T, paths.n_weights)
    for p in eachindex(paths.k)
        k1, k2, k3 = paths.k[p]
        nch = paths.n_weights_per_path[p]
        woff = paths.weight_offset[p]
        rng = (cg.poff[p] + 1):cg.poff[p + 1]
        for c in 1:nch
            wc = w[woff + c]
            acc_w = zero(T)
            @inbounds for t in rng
                m1 = Int(cg.m1[t]); m2 = Int(cg.m2[t]); m3 = Int(cg.m3[t]); v = cg.val[t]
                xi = x[_idx(in1, k1, c, m1)]
                yi = y[_idx(in2, k2, 1, m2)]
                gi = ḡ[_idx(out, k3, c, m3)]
                x̄[_idx(in1, k1, c, m1)] += wc * v * yi * gi
                ȳ[_idx(in2, k2, 1, m2)] += wc * v * xi * gi
                acc_w += v * xi * yi * gi
            end
            w̄[woff + c] += acc_w
        end
    end
    return x̄, ȳ, w̄
end
