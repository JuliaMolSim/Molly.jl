# Equivariant linear layer (channel mixing) and its analytic VJP.
#
# Per irrep entry `k` (fixed l, parity), a weight matrix `W_k ∈ R^{c_out × c_in}` mixes channels
# and is **shared across the 2l+1 components** — this sharing is exactly what preserves
# equivariance. A bias is allowed only on scalar (l=0, even) entries.
#
#   y[k, c_out, m] = Σ_{c_in} W_k[c_out, c_in] · x[k, c_in, m]   (+ b_k[c_out] if scalar)
#
# Pure maths, no Lux/HDF5 — core Molly. Internal (unexported).

"""
    EquivariantLinear(irreps_in, irreps_out, weights, biases)

An equivariant linear map. `irreps_in` and `irreps_out` must list the same irreps in the same
order (channel counts may differ). `weights[k]` is the `(c_out × c_in)` matrix for entry `k`;
`biases[k]` is a length-`c_out` vector (only used, and only nonzero, for scalar even entries).
"""
struct EquivariantLinear{T}
    irreps_in::Irreps
    irreps_out::Irreps
    weights::Vector{Matrix{T}}
    biases::Vector{Vector{T}}
    # Explicit inner constructor (with validation) so Julia does not auto-generate an outer
    # constructor that would clash with the convenience method below.
    function EquivariantLinear{T}(irreps_in::Irreps, irreps_out::Irreps,
                                  weights::Vector{Matrix{T}}, biases::Vector{Vector{T}}) where T
        length(irreps_in) == length(irreps_out) ||
            throw(ArgumentError("in/out irreps must have the same number of entries"))
        for k in 1:length(irreps_in)
            irreps_in.entries[k].ir == irreps_out.entries[k].ir ||
                throw(ArgumentError("entry $k irrep mismatch between in and out"))
            size(weights[k]) == (irreps_out.entries[k].mul, irreps_in.entries[k].mul) ||
                throw(ArgumentError("weight $k shape mismatch"))
        end
        return new{T}(irreps_in, irreps_out, weights, biases)
    end
end

EquivariantLinear(irreps_in::Irreps, irreps_out::Irreps,
                  weights::Vector{Matrix{T}}, biases::Vector{Vector{T}}) where T =
    EquivariantLinear{T}(irreps_in, irreps_out, weights, biases)

@inline _lin_idx(irs::Irreps, k::Int, c::Int, m::Int) =
    irs.offsets[k] + (c - 1) * (2 * irs.entries[k].ir.l + 1) + m

"""
    eqlinear_forward(lin, x) -> y

Apply the equivariant linear map to a flat input feature `x` (`dim(irreps_in)`), returning `y`
(`dim(irreps_out)`).
"""
function eqlinear_forward(lin::EquivariantLinear{T}, x::AbstractVector{T}) where T
    irs_i, irs_o = lin.irreps_in, lin.irreps_out
    y = zeros(T, irs_o.dim)
    for k in 1:length(irs_i)
        W = lin.weights[k]
        cout, cin = size(W)
        d = 2 * irs_i.entries[k].ir.l + 1
        is_scalar = irs_i.entries[k].ir.l == 0 && irs_i.entries[k].ir.p == 1
        for m in 1:d
            for co in 1:cout
                acc = zero(T)
                @inbounds for ci in 1:cin
                    acc += W[co, ci] * x[_lin_idx(irs_i, k, ci, m)]
                end
                if is_scalar
                    acc += lin.biases[k][co]
                end
                y[_lin_idx(irs_o, k, co, m)] = acc
            end
        end
    end
    return y
end

"""
    eqlinear_vjp(lin, x, ḡ) -> (x̄, W̄, b̄)

Reverse pass of the equivariant linear map. `ḡ = ∂E/∂y`; returns the input adjoint `x̄`, the
per-entry weight adjoints `W̄`, and bias adjoints `b̄` (nonzero only on scalar entries).
"""
function eqlinear_vjp(lin::EquivariantLinear{T}, x::AbstractVector{T}, ḡ::AbstractVector{T}) where T
    irs_i, irs_o = lin.irreps_in, lin.irreps_out
    x̄ = zeros(T, irs_i.dim)
    W̄ = [zeros(T, size(W)) for W in lin.weights]
    b̄ = [zeros(T, length(b)) for b in lin.biases]
    for k in 1:length(irs_i)
        W = lin.weights[k]
        cout, cin = size(W)
        d = 2 * irs_i.entries[k].ir.l + 1
        is_scalar = irs_i.entries[k].ir.l == 0 && irs_i.entries[k].ir.p == 1
        for m in 1:d
            for co in 1:cout
                g = ḡ[_lin_idx(irs_o, k, co, m)]
                if is_scalar
                    b̄[k][co] += g
                end
                @inbounds for ci in 1:cin
                    xi = x[_lin_idx(irs_i, k, ci, m)]
                    x̄[_lin_idx(irs_i, k, ci, m)] += W[co, ci] * g
                    W̄[k][co, ci] += g * xi
                end
            end
        end
    end
    return x̄, W̄, b̄
end
