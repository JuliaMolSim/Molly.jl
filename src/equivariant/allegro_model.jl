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
    # Independent per centre atom, so threaded over i (each atom writes only its own edge data).
    Threads.@threads for i in 1:n
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
        Threads.@threads for i in 1:n
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
        # per-edge update using the central atom's environment (independent per centre atom)
        Threads.@threads for i in 1:n
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

    Eatom = zeros(T, n)
    Threads.@threads for i in 1:n
        s = zero(T)
        for pos in eachindex(xs[i])
            s += (m.out_W * xs[i][pos] .+ m.out_b)[1]
        end
        Eatom[i] = s
    end
    return sum(Eatom)
end

"""
    allegro_energy_and_forces(m, coords, species, boundary, r_c) -> (E, F)

Total many-body energy `E` and analytic forces `F = -∂E/∂r`, via a hand-written reverse pass
(a full backward through every layer and the per-atom environment pooling). This mirrors Molly's
native ANI implementation: a per-atom descriptor pooled over neighbours whose backward scatters
`∂E/∂r` to every neighbour. The forward is taped per layer; the backward accumulates, per edge,
the adjoints of the initial equivariant/scalar latents and of the spherical harmonics and radial
cutoff, then converts those to Cartesian gradients through the SH Jacobian and the `|r|` chain.
"""
function allegro_energy_and_forces(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                                   species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    n = length(coords)
    nbr = neighbour_lists(coords, boundary, T(r_c))
    C, H, L = m.C, m.H, m.L
    dims = (1, 3, 5)
    shoff = ntuple(k -> m.feat.offsets[k] ÷ C, 3)   # SH block offset in a single-channel 9-vec
    foff  = ntuple(k -> m.feat.offsets[k], 3)        # feat block offset (channel-major)

    # ---- two-body precompute (per edge), plus geometry needed by the backward ----
    Ys  = [Vector{Vector{T}}() for _ in 1:n]
    us  = [T[] for _ in 1:n]
    ds  = [T[] for _ in 1:n]
    rhs = [SVector{3,T}[] for _ in 1:n]
    jjs = [Int[] for _ in 1:n]
    x0  = [Vector{Vector{T}}() for _ in 1:n]
    V0  = [Vector{Vector{T}}() for _ in 1:n]
    for i in 1:n
        for (j, d, rhat) in nbr[i]
            Y = collect(real_sph_harm(2, rhat))
            u = poly_envelope(d, m.r_c, m.env_p)
            R = bessel_basis(d, m.r_c, Val(m.nb)) .* u
            xe = dense_forward(m.emb_W2, m.emb_b2, silu.(dense_forward(m.emb_W1, m.emb_b1,
                     two_body_input(m, R, Int(species[i]), Int(species[j])))))
            push!(Ys[i], Y); push!(us[i], u); push!(ds[i], d); push!(rhs[i], rhat); push!(jjs[i], j)
            push!(x0[i], xe); push!(V0[i], init_equivariant(m, Y, u))
        end
    end

    # ---- forward with a per-layer tape ----
    xin = [[copy(x0[i][p]) for p in eachindex(x0[i])] for i in 1:n]
    Vin = [[copy(V0[i][p]) for p in eachindex(V0[i])] for i in 1:n]
    xin_tape = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], layer-input scalar latent
    Vin_tape = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], layer-input equivariant latent
    w_tape   = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], tp weights
    P_tape   = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], tp output
    a_tape   = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], resnet pre-activation
    Env_tape = Vector{Vector{Vector{T}}}(undef, L)           # [L][i]
    g_tape   = Vector{Vector{Vector{Vector{T}}}}(undef, L)   # [L][i][p], env weights (3C)

    for lidx in 1:L
        lw = m.layers[lidx]
        xin_tape[lidx] = [[copy(xin[i][p]) for p in eachindex(xin[i])] for i in 1:n]
        Vin_tape[lidx] = [[copy(Vin[i][p]) for p in eachindex(Vin[i])] for i in 1:n]
        Env = [zeros(T, m.feat.dim) for _ in 1:n]
        gL  = [Vector{Vector{T}}() for _ in 1:n]
        for i in 1:n
            e = Env[i]
            for p in eachindex(xin[i])
                g = dense_forward(lw.env_W, lw.env_b, xin[i][p])
                push!(gL[i], g)
                Y = Ys[i][p]
                @inbounds for k in 1:3
                    dk = dims[k]
                    for c in 1:C
                        gcl = g[(k - 1) * C + c]
                        base = foff[k] + (c - 1) * dk
                        for mm in 1:dk
                            e[base + mm] += gcl * Y[shoff[k] + mm]
                        end
                    end
                end
            end
            e ./= m.avg_nn
        end
        Env_tape[lidx] = Env
        g_tape[lidx] = gL
        wL = [Vector{Vector{T}}() for _ in 1:n]
        PL = [Vector{Vector{T}}() for _ in 1:n]
        aL = [Vector{Vector{T}}() for _ in 1:n]
        for i in 1:n
            for p in eachindex(xin[i])
                x = xin[i][p]
                w = dense_forward(lw.tp_W, lw.tp_b, x)
                P = tensor_product(m.paths, m.cg, Vin[i][p], Env[i], w)
                a = dense_forward(lw.x_W, lw.x_b, vcat(x, @view P[1:C]))
                xin[i][p] = x .+ silu.(a) .* us[i][p]
                Vin[i][p] = eqlinear_forward(lw.lin, P)
                push!(wL[i], w); push!(PL[i], P); push!(aL[i], a)
            end
        end
        w_tape[lidx] = wL; P_tape[lidx] = PL; a_tape[lidx] = aL
    end

    E = zero(T)
    for i in 1:n, p in eachindex(xin[i])
        E += (m.out_W * xin[i][p] .+ m.out_b)[1]
    end

    # ---- backward ----
    # x̄/V̄ are the adjoints of the CURRENT layer's edge outputs; Ȳ/ū accumulate geometry adjoints
    # across all layers (Y and u are reused every layer, in the env pooling and the resnet cutoff).
    xbar = [[collect(m.out_W[1, :]) for _ in eachindex(xin[i])] for i in 1:n]   # ∂E/∂x_final
    Vbar = [[zeros(T, m.feat.dim) for _ in eachindex(xin[i])] for i in 1:n]     # ∂E/∂V_final = 0
    Ybar = [[zeros(T, 9) for _ in eachindex(xin[i])] for i in 1:n]
    ubar = [zeros(T, length(xin[i])) for i in 1:n]

    for lidx in L:-1:1
        lw = m.layers[lidx]
        Env = Env_tape[lidx]
        xin_bar = [[zeros(T, H) for _ in eachindex(xin[i])] for i in 1:n]
        Vin_bar = [[zeros(T, m.feat.dim) for _ in eachindex(xin[i])] for i in 1:n]
        Envbar = [zeros(T, m.feat.dim) for _ in 1:n]
        # Pass A: backward of each edge's update
        for i in 1:n
            for p in eachindex(xin[i])
                xin_p = xin_tape[lidx][i][p]
                Vin_p = Vin_tape[lidx][i][p]
                w = w_tape[lidx][i][p]
                P = P_tape[lidx][i][p]
                a = a_tape[lidx][i][p]
                u = us[i][p]
                xb = xbar[i][p]
                Vb = Vbar[i][p]
                Pbar = zeros(T, m.feat.dim)
                # V_out = eqlinear(lin, P)
                Px, _, _ = eqlinear_vjp(lw.lin, P, Vb)
                Pbar .+= Px
                # x_out = x_in + silu(a)·u
                xin_bar[i][p] .+= xb                          # identity residual
                sa = silu.(a)
                da = xb .* silu_grad.(a) .* u                 # ∂E/∂a
                acc_u = zero(T)
                @inbounds for q in eachindex(xb)
                    acc_u += xb[q] * sa[q]
                end
                ubar[i][p] += acc_u                           # ∂E/∂u from the resnet
                # a = x_W·[x_in; scalars_0e(P)] + x_b
                gin = transpose(lw.x_W) * da                  # length H+C
                @views xin_bar[i][p] .+= gin[1:H]
                @views Pbar[1:C] .+= gin[H + 1:H + C]
                # P = TP(V_in, Env_i, w)
                Vx, Ex, wbar = tensor_product_vjp(m.paths, m.cg, Vin_p, Env[i], w, Pbar)
                Vin_bar[i][p] .+= Vx
                Envbar[i] .+= Ex
                # w = tp_W·x_in + tp_b
                xin_bar[i][p] .+= transpose(lw.tp_W) * wbar
            end
        end
        # Pass B: backward of the environment pooling, scattering Envbar to each neighbour edge
        invavg = one(T) / m.avg_nn
        for i in 1:n
            Eb = Envbar[i]
            for p in eachindex(xin[i])
                g = g_tape[lidx][i][p]
                Y = Ys[i][p]
                gbar = zeros(T, 3 * C)
                @inbounds for k in 1:3
                    dk = dims[k]
                    for c in 1:C
                        base = foff[k] + (c - 1) * dk
                        gcl = g[(k - 1) * C + c]
                        acc_g = zero(T)
                        for mm in 1:dk
                            eb = Eb[base + mm] * invavg
                            acc_g += eb * Y[shoff[k] + mm]
                            Ybar[i][p][shoff[k] + mm] += eb * gcl
                        end
                        gbar[(k - 1) * C + c] = acc_g
                    end
                end
                # g = env_W·x_in + env_b
                xin_bar[i][p] .+= transpose(lw.env_W) * gbar
            end
        end
        xbar = xin_bar
        Vbar = Vin_bar
    end

    # ---- backprop the two-body / initial-latent chain to Cartesian gradients ----
    dEdc = [zero(SVector{3,T}) for _ in 1:n]
    for i in 1:n
        for p in eachindex(xin[i])
            d = ds[i][p]; rhat = rhs[i][p]; u = us[i][p]; Y = Ys[i][p]; j = jjs[i][p]
            Yb = Ybar[i][p]; ub = ubar[i][p]; Vb0 = Vbar[i][p]; xb0 = xbar[i][p]
            # V^0 = init_equivariant(Y, u): V0[k,c,m] = init_w[c,k]·Y[k,m]·u (+ init_b0[c]·u on 0e)
            @inbounds for k in 1:3
                dk = dims[k]
                for c in 1:C
                    base = foff[k] + (c - 1) * dk
                    iw = m.init_w[c, k]
                    for mm in 1:dk
                        vb = Vb0[base + mm]
                        Yb[shoff[k] + mm] += iw * u * vb
                        ub += iw * Y[shoff[k] + mm] * vb
                    end
                end
            end
            @inbounds for c in 1:C
                ub += m.init_b0[c] * Vb0[foff[1] + c]
            end
            # x^0 = emb_W2·silu(emb_W1·s_in + emb_b1) + emb_b2  (recompute the forward)
            B, dB = bessel_basis_grad(d, m.r_c, Val(m.nb))
            _, du = poly_envelope_grad(d, m.r_c, m.env_p)
            R = B .* u
            s_in = two_body_input(m, R, Int(species[i]), j <= 0 ? 1 : Int(species[j]))
            h1 = dense_forward(m.emb_W1, m.emb_b1, s_in)
            da1 = transpose(m.emb_W2) * xb0
            dh1 = da1 .* silu_grad.(h1)
            ds_in = transpose(m.emb_W1) * dh1
            # R = B·u  (both B and u depend on d)
            dEdd = zero(T)
            @inbounds for q in 1:m.nb
                dR = ds_in[q]
                ub += dR * B[q]           # ∂E/∂u via R
                dEdd += dR * dB[q] * u    # ∂E/∂d via the Bessel basis
            end
            dEdd += ub * du               # ∂E/∂d via the polynomial envelope
            # geometry: Y depends on r̂ (tangential, via the SH Jacobian); u, R depend on d = |r|
            r_raw = rhat * d
            _, J = real_sph_harm_grad(2, r_raw)   # J[q, b] = ∂Y[q]/∂r_b
            gx = dEdd * rhat[1]; gy = dEdd * rhat[2]; gz = dEdd * rhat[3]
            @inbounds for q in 1:9
                gx += J[q, 1] * Yb[q]; gy += J[q, 2] * Yb[q]; gz += J[q, 3] * Yb[q]
            end
            dEdr = SVector{3,T}(gx, gy, gz)      # r = coords[j] - coords[i]
            dEdc[j] += dEdr
            dEdc[i] -= dEdr
        end
    end
    F = [-dEdc[i] for i in 1:n]
    return E, F
end

"""
    allegro_forces(m, coords, species, boundary, r_c) -> Vector{SVector{3,T}}

Analytic forces `F = -∂E/∂r` via [`allegro_energy_and_forces`](@ref).
"""
function allegro_forces(m::AllegroModel{T}, coords::AbstractVector{<:SVector{3}},
                        species::AbstractVector{<:Integer}, boundary, r_c::T) where T
    _, F = allegro_energy_and_forces(m, coords, species, boundary, T(r_c))
    return F
end
