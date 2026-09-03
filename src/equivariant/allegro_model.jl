# Allegro-style equivariant energy model (CPU reference forward), built on the equivariant
# primitives. Strictly local: the energy is a sum over directed edges within a cutoff. This holds
# the pure maths (no Lux/HDF5) so it can be tested with a bare `using Molly`; the HDF5 weight
# loading and AtomsCalculators wiring live in the extension.
#
# Per directed edge i<-j (i = central atom), with r = r_j - r_i, d = |r|, r̂ = r/d:
#   Y = real_sph_harm(2, r̂)              (9,)          u = poly_envelope(d)
#   R = bessel(d)·u                        (nb,)
#   s_in = [R; onehot(Z_i); onehot(Z_j)]
#   x = MLP_embed(s_in)                    (H,)  scalar latent
#   V = init_lin(Y)·u                      (9C,) equivariant latent, C×(0e+1o+2e)
#   for each layer:
#       w = x·tp_W' + tp_b                 tensor-product path weights
#       P = TP_uvu(V, Y; w)                (9C,)
#       x = x + silu([x; scalars0e(P)]·x_W' + x_b)
#       V = eqlin(P)
#   E_edge = out_W·x + out_b               (scalar)
# The activation is SiLU; the tensor product uses e3nn-normalized (wigner_3j) coefficients.

"""
    AllegroModel

A loaded Allegro-style model: hyperparameters, the precomputed tensor-product paths and
coefficients, and the per-layer weights. Construct with [`build_allegro_model`](@ref). The energy
forward is [`allegro_edge_energy`](@ref) / [`allegro_total_energy`](@ref).
"""
struct AllegroModel{T}
    C::Int; H::Int; nb::Int; S::Int; L::Int; env_p::Int
    r_c::T
    feat::Irreps
    sh::Irreps
    paths::TensorProductPaths
    cg::SparseCG{T}
    emb_W1::Matrix{T}; emb_b1::Vector{T}
    emb_W2::Matrix{T}; emb_b2::Vector{T}
    init_w::Matrix{T}   # (C, 3): per-l channel scale (columns are l = 0,1,2)
    init_b0::Vector{T}  # (C,) scalar bias applied to the 0e block at init
    layers::Vector{NamedTuple{(:tp_W, :tp_b, :x_W, :x_b, :lin),
                              Tuple{Matrix{T}, Vector{T}, Matrix{T}, Vector{T}, EquivariantLinear{T}}}}
    out_W::Matrix{T}; out_b::Vector{T}
end

"""
    build_allegro_model(; C, H, nb, S, L, env_p, r_c, weights, T=Float64)

Assemble an [`AllegroModel`](@ref) from hyperparameters and a `weights` NamedTuple (see the field
comments). Builds the `feat ⊗ sh → feat` tensor-product paths (uvu) and their e3nn-normalized
Clebsch-Gordan table. `lmax` is fixed at 2.
"""
function build_allegro_model(; C::Int, H::Int, nb::Int, S::Int, L::Int, env_p::Int, r_c::Real,
                             weights, T::Type=Float64)
    feat = Irreps("$(C)x0e+$(C)x1o+$(C)x2e")
    sh = Irreps("1x0e+1x1o+1x2e")
    paths = TensorProductPaths(feat, sh, feat)
    cg = build_sparse_cg(paths; T=T, normalization=:wigner3j)
    layers = map(weights.layers) do lw
        # equivariant linear feat→feat: per-l C×C weight, bias only on the 0e block
        Wl = [T.(lw.lin_w[:, :, k]) for k in 1:3]
        bl = [k == 1 ? T.(lw.lin_b0) : zeros(T, C) for k in 1:3]
        lin = EquivariantLinear(feat, feat, Wl, bl)
        (tp_W=T.(lw.tp_W), tp_b=T.(lw.tp_b), x_W=T.(lw.x_W), x_b=T.(lw.x_b), lin=lin)
    end
    return AllegroModel{T}(C, H, nb, S, L, env_p, T(r_c), feat, sh, paths, cg,
                           T.(weights.emb_W1), T.(weights.emb_b1), T.(weights.emb_W2), T.(weights.emb_b2),
                           T.(weights.init_w), T.(weights.init_b0),
                           layers, T.(weights.out_W), T.(weights.out_b))
end

@inline _dense(W, b, x) = W * x .+ b

# derivative of SiLU: d/dx[x·σ(x)] = σ(x)·(1 + x·(1−σ(x)))
@inline function _silu_grad(x::T) where T
    s = one(T) / (one(T) + exp(-x))
    return s * (one(T) + x * (one(T) - s))
end

# scalar (0e) block of a feature vector: the first C entries (entry k=1 is the 0e block, mul C)
@inline _scalars0e(feat::Irreps, P, C) = @view P[1:C]

"""
    allegro_edge_energy(m::AllegroModel, d, rhat, Zi, Zj) -> T

Energy contribution of one directed edge with length `d`, unit direction `rhat` (`SVector{3}`),
central species index `Zi` and neighbour species `Zj` (both 1-based). Returns 0 if `d ≥ r_c`.
"""
function allegro_edge_energy(m::AllegroModel{T}, d::T, rhat::SVector{3,T}, Zi::Int, Zj::Int) where T
    d < m.r_c || return zero(T)
    Y = real_sph_harm(2, rhat)                       # SVector length 9
    u = poly_envelope(d, m.r_c, m.env_p)
    R = bessel_basis(d, m.r_c, Val(m.nb)) .* u        # radial embedding
    # two-body scalar input [R; onehot(Zi); onehot(Zj)]
    s_in = zeros(T, m.nb + 2 * m.S)
    @inbounds for i in 1:m.nb; s_in[i] = R[i]; end
    s_in[m.nb + Zi] = one(T)
    s_in[m.nb + m.S + Zj] = one(T)
    x = _dense(m.emb_W2, m.emb_b2, silu.(_dense(m.emb_W1, m.emb_b1, s_in)))  # (H,)

    # initial equivariant latent V = init_lin(Y)·u, channel-major
    V = zeros(T, m.feat.dim)
    @inbounds for k in 1:3
        d_k = 2 * (k - 1) + 1                          # 1,3,5
        yoff = m.sh.offsets[k]
        for c in 1:m.C
            wl = m.init_w[c, k]
            base = m.feat.offsets[k] + (c - 1) * d_k
            for mm in 1:d_k
                V[base + mm] = wl * Y[yoff + mm] * u
            end
        end
    end
    @inbounds for c in 1:m.C
        V[m.feat.offsets[1] + c] += m.init_b0[c] * u   # 0e bias
    end

    Yv = collect(Y)  # tensor_product expects an AbstractVector for in2
    for lw in m.layers
        w = _dense(lw.tp_W, lw.tp_b, x)                 # (n_weights,)
        P = tensor_product(m.paths, m.cg, V, Yv, w)     # (9C,)
        scal = _scalars0e(m.feat, P, m.C)
        x = x .+ silu.(_dense(lw.x_W, lw.x_b, vcat(x, scal)))
        V = eqlinear_forward(lw.lin, P)
    end
    return (m.out_W * x .+ m.out_b)[1]
end

"""
    allegro_edge_energy_and_grad(m, d, rhat, Zi, Zj) -> (E, g)

Energy of one directed edge and its analytic gradient `g = ∂E_edge/∂r` with respect to the edge
vector `r = d·rhat` (`SVector{3}`), obtained by composing the primitive VJPs. Returns `(0, 0)` if
`d ≥ r_c`.
"""
function allegro_edge_energy_and_grad(m::AllegroModel{T}, d::T, rhat::SVector{3,T},
                                      Zi::Int, Zj::Int) where T
    (d < m.r_c) || return (zero(T), zero(SVector{3,T}))
    rvec = d .* rhat
    Y, JY = real_sph_harm_grad(2, rvec)                 # Y (9,), JY = ∂Y/∂r (9×3)
    Yv = collect(Y)
    u, du = poly_envelope_grad(d, m.r_c, m.env_p)
    B, dB = bessel_basis_grad(d, m.r_c, Val(m.nb))
    R = B .* u

    # ---- forward, caching activations ----
    s_in = zeros(T, m.nb + 2 * m.S)
    @inbounds for i in 1:m.nb; s_in[i] = R[i]; end
    s_in[m.nb + Zi] = one(T); s_in[m.nb + m.S + Zj] = one(T)
    a1 = _dense(m.emb_W1, m.emb_b1, s_in)
    x0 = _dense(m.emb_W2, m.emb_b2, silu.(a1))

    V0 = zeros(T, m.feat.dim)
    @inbounds for k in 1:3
        dk = 2 * (k - 1) + 1; yoff = m.sh.offsets[k]
        for c in 1:m.C
            wl = m.init_w[c, k]; base = m.feat.offsets[k] + (c - 1) * dk
            for mm in 1:dk; V0[base + mm] = wl * Y[yoff + mm] * u; end
        end
    end
    @inbounds for c in 1:m.C; V0[m.feat.offsets[1] + c] += m.init_b0[c] * u; end

    L = length(m.layers)
    x_ins = Vector{Vector{T}}(undef, L); V_ins = Vector{Vector{T}}(undef, L)
    ws = Vector{Vector{T}}(undef, L); Ps = Vector{Vector{T}}(undef, L); axs = Vector{Vector{T}}(undef, L)
    x = x0; V = V0
    for (li, lw) in enumerate(m.layers)
        x_ins[li] = x; V_ins[li] = V
        w = _dense(lw.tp_W, lw.tp_b, x); ws[li] = w
        P = tensor_product(m.paths, m.cg, V, Yv, w); Ps[li] = P
        scal = P[1:m.C]
        ax = _dense(lw.x_W, lw.x_b, vcat(x, scal)); axs[li] = ax
        x = x .+ silu.(ax)
        V = eqlinear_forward(lw.lin, P)
    end
    E = (m.out_W * x .+ m.out_b)[1]

    # ---- backward ----
    gx = vec(collect(m.out_W[1, :]))                    # ∂E/∂x_final
    gV = zeros(T, m.feat.dim)                           # ∂E/∂V_final (unused by readout)
    gY = zeros(T, length(Yv))                           # ∂E/∂Y (accumulated)
    for li in L:-1:1
        lw = m.layers[li]
        gh = gx                                         # residual: x_out = x_in + silu(ax)
        gax = gh .* _silu_grad.(axs[li])
        gcat = lw.x_W' * gax                            # ∂E/∂[x_in; scal]
        gx_in = copy(gx)                                # residual path
        @inbounds for i in 1:m.H; gx_in[i] += gcat[i]; end
        gscal = @view gcat[(m.H + 1):(m.H + m.C)]
        gP = zeros(T, m.feat.dim)
        @inbounds for c in 1:m.C; gP[c] += gscal[c]; end
        gP_lin, _, _ = eqlinear_vjp(lw.lin, Ps[li], gV)
        gP .+= gP_lin
        gVin, gYc, gw = tensor_product_vjp(m.paths, m.cg, V_ins[li], Yv, ws[li], gP)
        gY .+= gYc
        gx_in .+= lw.tp_W' * gw
        gx = gx_in; gV = gVin
    end
    # gx = ∂E/∂x0 ; gV = ∂E/∂V0
    gu = zero(T)
    @inbounds for k in 1:3
        dk = 2 * (k - 1) + 1; yoff = m.sh.offsets[k]
        for c in 1:m.C
            wl = m.init_w[c, k]; base = m.feat.offsets[k] + (c - 1) * dk
            for mm in 1:dk
                gY[yoff + mm] += gV[base + mm] * wl * u
                gu += gV[base + mm] * wl * Y[yoff + mm]
            end
        end
    end
    @inbounds for c in 1:m.C; gu += gV[m.feat.offsets[1] + c] * m.init_b0[c]; end
    # embedding backward → ∂E/∂R
    g_silu_a1 = m.emb_W2' * gx
    ga1 = g_silu_a1 .* _silu_grad.(a1)
    gs_in = m.emb_W1' * ga1
    # radial ∂E/∂d : R_n = B_n·u  ⇒ dR_n/dd = dB_n·u + B_n·du ; plus u's direct role in V init
    gd = zero(T)
    @inbounds for n in 1:m.nb
        gd += gs_in[n] * (dB[n] * u + B[n] * du)
    end
    gd += gu * du
    # assemble ∂E/∂r = JYᵀ·gY (directional) + gd·rhat (radial)
    gr = MVector{3,T}(0, 0, 0)
    @inbounds for b in 1:3
        acc = zero(T)
        for i in 1:length(Yv); acc += JY[i, b] * gY[i]; end
        gr[b] = acc + gd * rhat[b]
    end
    return E, SVector{3,T}(gr)
end

"""
    allegro_forces(m, coords, species, boundary, r_c) -> Vector{SVector{3,T}}

Analytic forces `F = -∂E/∂r` for all atoms, summing each directed edge's contribution: for edge
`i ← j` with `g = ∂E_edge/∂r`, `F[i] += g` and `F[j] -= g`. Also returns nothing extra; use
[`allegro_total_energy`](@ref) for the energy.
"""
function allegro_forces(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                        species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    n = length(coords)
    F = [zero(SVector{3,T}) for _ in 1:n]
    for i in 1:n, j in 1:n
        i == j && continue
        r = isnothing(boundary) ? (coords[j] - coords[i]) : vector(coords[i], coords[j], boundary)
        d = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
        (d < r_c && d > 1e-8) || continue
        _, g = allegro_edge_energy_and_grad(m, T(d), r ./ T(d), Int(species[i]), Int(species[j]))
        F[i] += g
        F[j] -= g
    end
    return F
end

"""
    allegro_total_energy(m, coords, species, boundary, r_c) -> T

Total energy: sum of [`allegro_edge_energy`](@ref) over all ordered pairs (i, j) with
`0 < |r_ij| < r_c`, using the minimum-image displacement under `boundary` (`nothing` ⇒ no PBC).
`coords` are `SVector{3}` in the model's length unit (Å); `species` are 1-based indices.
"""
function allegro_total_energy(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                              species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    n = length(coords)
    E = zero(T)
    for i in 1:n, j in 1:n
        i == j && continue
        r = isnothing(boundary) ? (coords[j] - coords[i]) : vector(coords[i], coords[j], boundary)
        d = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
        (d < r_c && d > 1e-8) || continue
        E += allegro_edge_energy(m, T(d), r ./ T(d), Int(species[i]), Int(species[j]))
    end
    return E
end
