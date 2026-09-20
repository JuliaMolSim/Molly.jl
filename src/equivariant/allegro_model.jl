# Allegro-style equivariant energy model (CPU reference forward). Strictly local and **many-body**:
# each layer's tensor product multiplies an edge's equivariant latent by the CENTRAL ATOM'S
# ENVIRONMENT — a sum over all of that atom's neighbours — not the edge's own spherical harmonics.
# This is what makes it Allegro rather than a pair potential. Pure maths (no HDF5); the HDF5 weight
# loading lives in the extension.
#
# Per atom i, per directed edge i<-j (r = r_j - r_i, d = |r|, r̂ = r/d):
#   Y_ij = real_sph_harm(2, r̂);  u_ij = poly_envelope(d);  R_ij = bessel(d)·u_ij
#   x_ij^0 = MLP_emb([R_ij, onehot(Z_i), onehot(Z_j)])                    # scalar latent (H)
#   V_ij^0 = init_lin(Y_ij)·u_ij                                          # equivariant latent, C×(0e+1o+2e)
#   for each layer L:
#     g_ik   = env_W^L·x_ik + env_b^L                                     # per (l,channel) env weights (3C)
#     Env_i  = (1/avg_nn) Σ_{k∈N(i)} g_ik[l,c] · Y_ik[l,·]                # C-channel environment (feat)
#     P_ij   = TP_uvu(V_ij, Env_i; w_ij),  w_ij = tp_W^L·x_ij + tp_b^L    # feat ⊗ feat → feat
#     x_ij   = x_ij + silu(x_W^L·[x_ij, scalars_0e(P_ij)] + x_b^L)·u_ij   # residual + cutoff
#     V_ij   = eqlinear^L(P_ij)
#   E_ij = out_W·x_ij + out_b;   E = Σ_i Σ_{j∈N(i)} E_ij

"""
    AllegroModel

A loaded many-body Allegro model: hyperparameters, precomputed `feat ⊗ feat → feat` tensor-product
paths/coefficients, and per-layer weights (including the environment-embedding weights). Construct
with [`build_allegro_model`](@ref); the energy forward is [`allegro_total_energy`](@ref).
"""
struct AllegroModel{T}
    C::Int
    H::Int
    nb::Int
    S::Int
    L::Int
    env_p::Int
    r_c::T
    avg_nn::T
    feat::Irreps
    paths::TensorProductPaths
    cg::SparseCG{T}
    emb_W1::Matrix{T}
    emb_b1::Vector{T}
    emb_W2::Matrix{T}
    emb_b2::Vector{T}
    init_w::Matrix{T}   # (C, 3): per-l channel scale for the initial equivariant latent
    init_b0::Vector{T}  # (C,) bias on the 0e block at init
    layers::Vector{NamedTuple{(:env_W, :env_b, :tp_W, :tp_b, :x_W, :x_b, :lin),
                              Tuple{Matrix{T}, Vector{T}, Matrix{T}, Vector{T}, Matrix{T},
                                    Vector{T}, EquivariantLinear{T}}}}
    out_W::Matrix{T}
    out_b::Vector{T}
end

"""
    build_allegro_model(; C, H, nb, S, L, env_p, r_c, avg_nn, weights, T=Float64)

Assemble an [`AllegroModel`](@ref). Builds the `feat ⊗ feat → feat` uvu tensor-product paths and
their e3nn-normalised Clebsch-Gordan table. `lmax` is fixed at 2.
"""
function build_allegro_model(; C::Int, H::Int, nb::Int, S::Int, L::Int, env_p::Int, r_c::Real,
                             avg_nn::Real, weights, T::Type=Float64)
    feat = Irreps("$(C)x0e+$(C)x1o+$(C)x2e")
    paths = TensorProductPaths(feat, feat, feat)
    cg = build_sparse_cg(paths; T=T, normalization=:wigner3j)
    layers = map(weights.layers) do lw
        Wl = [T.(lw.lin_w[:, :, k]) for k in 1:3]
        bl = [k == 1 ? T.(lw.lin_b0) : zeros(T, C) for k in 1:3]
        lin = EquivariantLinear(feat, feat, Wl, bl)
        (env_W=T.(lw.env_W), env_b=T.(lw.env_b), tp_W=T.(lw.tp_W), tp_b=T.(lw.tp_b),
         x_W=T.(lw.x_W), x_b=T.(lw.x_b), lin=lin)
    end
    return AllegroModel{T}(C, H, nb, S, L, env_p, T(r_c), T(avg_nn), feat, paths, cg,
                           T.(weights.emb_W1), T.(weights.emb_b1), T.(weights.emb_W2), T.(weights.emb_b2),
                           T.(weights.init_w), T.(weights.init_b0), layers,
                           T.(weights.out_W), T.(weights.out_b))
end

@inline dense_forward(W, b, x) = W * x .+ b

@inline function silu_grad(x::T) where T
    s = one(T) / (one(T) + exp(-x))
    return s * (one(T) + x * (one(T) - s))
end

# Two-body scalar input [radial embedding; onehot(Zi); onehot(Zj)].
function two_body_input(m::AllegroModel{T}, R, Zi, Zj) where T
    s_in = zeros(T, m.nb + 2 * m.S)
    @inbounds for i in 1:m.nb
        s_in[i] = R[i]
    end
    s_in[m.nb + Zi] = one(T)
    s_in[m.nb + m.S + Zj] = one(T)
    return s_in
end

# Initial equivariant latent V_ij^0 = init_lin(Y)·u (channel-major feat vector).
function init_equivariant(m::AllegroModel{T}, Y, u) where T
    V = zeros(T, m.feat.dim)
    @inbounds for k in 1:3
        dk = 2 * (k - 1) + 1
        yoff = m.feat.offsets[k] ÷ m.C  # offset of block k in a single-channel (sh) 9-vector
        for c in 1:m.C
            wl = m.init_w[c, k]
            base = m.feat.offsets[k] + (c - 1) * dk
            for mm in 1:dk
                V[base + mm] = wl * Y[yoff + mm] * u
            end
        end
    end
    @inbounds for c in 1:m.C
        V[m.feat.offsets[1] + c] += m.init_b0[c] * u
    end
    return V
end

# Per-atom neighbour lists within the cutoff: returns nbr[i] = Vector of (j, d, r̂).
function neighbour_lists(coords::AbstractVector{<:SVector{3,T}}, boundary, r_c::T) where T
    n = length(coords)
    nbr = [Tuple{Int,T,SVector{3,T}}[] for _ in 1:n]
    for i in 1:n, j in 1:n
        i == j && continue
        r = isnothing(boundary) ? (coords[j] - coords[i]) : vector(coords[i], coords[j], boundary)
        d = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
        (d < r_c && d > 1e-8) || continue
        push!(nbr[i], (j, d, r ./ d))
    end
    return nbr
end

"""
    allegro_total_energy(m, coords, species, boundary, r_c) -> T

Total many-body Allegro energy: `Σ_i Σ_{j∈N(i)} E_ij`, with each edge coupled to the central
atom's environment. `coords` are `SVector{3}` in Å; `species` are 1-based.
"""
function allegro_total_energy(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                              species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    n = length(coords)
    nbr = neighbour_lists(coords, boundary, T(r_c))
    C = m.C
    # per-center-atom edge data (indexed [i][pos] over neighbours of i)
    Ys = [Vector{Vector{T}}() for _ in 1:n]   # single-channel SH (9-vec) per edge
    us = [T[] for _ in 1:n]
    xs = [Vector{Vector{T}}() for _ in 1:n]   # scalar latent per edge
    Vs = [Vector{Vector{T}}() for _ in 1:n]   # equivariant latent per edge (feat)
    for i in 1:n
        for (j, d, rhat) in nbr[i]
            Y = collect(real_sph_harm(2, rhat))
            u = poly_envelope(d, m.r_c, m.env_p)
            R = bessel_basis(d, m.r_c, Val(m.nb)) .* u
            x0 = dense_forward(m.emb_W2, m.emb_b2, silu.(dense_forward(m.emb_W1, m.emb_b1,
                     two_body_input(m, R, Int(species[i]), Int(species[j])))))
            push!(Ys[i], Y); push!(us[i], u)
            push!(xs[i], x0); push!(Vs[i], init_equivariant(m, Y, u))
        end
    end

    for lw in m.layers
        # environment per atom (density trick): Env_i = (1/avg_nn) Σ_k g_ik[l,c]·Y_ik
        Env = [zeros(T, m.feat.dim) for _ in 1:n]
        for i in 1:n
            e = Env[i]
            @inbounds for pos in eachindex(xs[i])
                g = dense_forward(lw.env_W, lw.env_b, xs[i][pos])   # length 3C, layout [l-major, channel]
                Y = Ys[i][pos]
                for k in 1:3
                    dk = 2 * (k - 1) + 1
                    yoff = m.feat.offsets[k] ÷ C
                    for c in 1:C
                        gcl = g[(k - 1) * C + c]
                        base = m.feat.offsets[k] + (c - 1) * dk
                        for mm in 1:dk
                            e[base + mm] += gcl * Y[yoff + mm]
                        end
                    end
                end
            end
            e ./= m.avg_nn
        end
        # per-edge update using the central atom's environment
        for i in 1:n
            for pos in eachindex(xs[i])
                x = xs[i][pos]
                w = dense_forward(lw.tp_W, lw.tp_b, x)
                P = tensor_product(m.paths, m.cg, Vs[i][pos], Env[i], w)
                scal = @view P[1:C]
                us_ip = us[i][pos]
                xs[i][pos] = x .+ silu.(dense_forward(lw.x_W, lw.x_b, vcat(x, scal))) .* us_ip
                Vs[i][pos] = eqlinear_forward(lw.lin, P)
            end
        end
    end

    E = zero(T)
    for i in 1:n
        for pos in eachindex(xs[i])
            E += (m.out_W * xs[i][pos] .+ m.out_b)[1]
        end
    end
    return E
end

"""
    allegro_forces(m, coords, species, boundary, r_c) -> Vector{SVector{3,T}}

Forces `F = -∂E/∂r`. **Temporary:** central finite differences of [`allegro_total_energy`](@ref);
the analytic ANI-style backward (with the environment gradient scattered to all neighbours) is the
next milestone.
"""
function allegro_forces(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                        species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    n = length(coords)
    F = Vector{SVector{3,T}}(undef, n)
    h = T(1e-5)
    cc = collect(SVector{3,T}, coords)
    for i in 1:n
        g = zero(MVector{3,T})
        for b in 1:3
            orig = cc[i]
            cc[i] = setindex(orig, orig[b] + h, b)
            Ep = allegro_total_energy(m, cc, species, boundary, T(r_c))
            cc[i] = setindex(orig, orig[b] - h, b)
            Em = allegro_total_energy(m, cc, species, boundary, T(r_c))
            cc[i] = orig
            g[b] = -(Ep - Em) / (2h)
        end
        F[i] = SVector{3,T}(g)
    end
    return F
end
