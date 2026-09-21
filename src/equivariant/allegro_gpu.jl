# GPU-portable Allegro energy forward (KernelAbstractions). Mirrors the CPU forward in
# allegro_model.jl but runs as a sequence of data-parallel kernels over a flat directed-edge
# layout, so the same code runs on the KA CPU backend, CUDA and Metal. Each kernel writes its
# outputs to global device arrays (one column per edge or per atom) — no per-thread dynamically
# sized scratch, which is what keeps it portable.
#
# Layout: build_neighbor_csr gives per-centre-atom neighbour offsets `off` (n+1) and neighbour
# atom indices `idx` (n_edges). Edge e belongs to centre atom `ecenter[e]` and points to atom
# `idx[e]`; its unit vector `rhat[:,e]` and distance `d[e]` are precomputed on the host (the CSR
# build is already an O(N²) host pass). KernelAbstractions is a core Molly dependency, so this
# lives in core (no extension needed); only loading the weights from HDF5 needs the extension.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

# Flat, device-resident form of an AllegroModel plus the CG/path tables. Built once per
# (model, backend, T) and reused across systems.
struct AllegroGPU{T, VT, MT, IT, VTI}
    C::Int; H::Int; nb::Int; S::Int; L::Int; env_p::Int
    fd::Int; nw::Int; np::Int
    o1::Int; o2::Int; o3::Int          # feat block offsets (channel-major)
    r_c::T; avg_nn::T
    # two-body / readout / init weights (device)
    emb_W1::MT; emb_b1::VT; emb_W2::MT; emb_b2::VT
    init_w::MT; init_b0::VT
    out_W::MT; out_b::VT
    # per-layer weights, stored as vectors of device arrays
    env_W::Vector{MT}; env_b::Vector{VT}
    tp_W::Vector{MT};  tp_b::Vector{VT}
    x_W::Vector{MT};   x_b::Vector{VT}
    lin_w::Vector{MT}; lin_b0::Vector{VT}   # lin_w[l] is (C, C*3) packed [block1|block2|block3]? no: (C,C) per block ⇒ store (C, 3C)
    # CG / path tables (device, Int32)
    p_k1::IT; p_k2::IT; p_k3::IT; p_woff::IT; poff::IT
    cg_m1::IT; cg_m2::IT; cg_m3::IT; cg_val::VTI
end

# feat flat index (channel-major), 1-based m; offsets passed explicitly.
@inline _foff(o1, o2, o3, k) = k == 1 ? o1 : (k == 2 ? o2 : o3)
@inline _fdim(k) = 2 * (k - 1) + 1
@inline _fidx(o1, o2, o3, k, c, m) = _foff(o1, o2, o3, k) + (c - 1) * _fdim(k) + m

# ---- kernels ---------------------------------------------------------------------------------

# Per edge: real SH (l≤2), polynomial envelope u, Bessel radial R = B·u.
@kernel inbounds=true function allegro_geom_kernel!(Y, u, R, @Const(rhx), @Const(rhy), @Const(rhz),
                                                    @Const(d), rc, env_p, nb)
    e = @index(Global, Linear)
    T = eltype(u)
    x = rhx[e]; y = rhy[e]; z = rhz[e]
    Yv = real_sph_harm(2, SVector{3,T}(x, y, z))
    for q in 1:9
        Y[q, e] = Yv[q]
    end
    de = d[e]
    ue = poly_envelope(de, rc, env_p)
    u[e] = ue
    s = sqrt(T(2) / rc)
    for q in 1:nb
        R[q, e] = s * sin(q * T(pi) * de / rc) / de * ue
    end
end

# Per edge: two-body scalar latent x^0 = emb_W2·silu(emb_W1·[R;1hot(Zi);1hot(Zj)]+emb_b1)+emb_b2.
@kernel inbounds=true function allegro_emb_kernel!(X, A1, @Const(R), @Const(spi), @Const(spj),
                                                   @Const(W1), @Const(b1), @Const(W2), @Const(b2),
                                                   nb, S, H)
    e = @index(Global, Linear)
    T = eltype(X)
    zi = spi[e]; zj = spj[e]
    for h in 1:H
        acc = b1[h]
        for q in 1:nb
            acc += W1[h, q] * R[q, e]
        end
        acc += W1[h, nb + zi]
        acc += W1[h, nb + S + zj]
        A1[h, e] = acc / (one(T) + exp(-acc))     # silu
    end
    for h in 1:H
        acc = b2[h]
        for k in 1:H
            acc += W2[h, k] * A1[k, e]
        end
        X[h, e] = acc
    end
end

# Per edge: initial equivariant latent V^0 = init_lin(Y)·u (+ init_b0 on 0e).
@kernel inbounds=true function allegro_init_kernel!(V, @Const(Y), @Const(u), @Const(iw), @Const(ib0),
                                                    C, o1, o2, o3)
    e = @index(Global, Linear)
    ue = u[e]
    for k in 1:3
        dk = _fdim(k)
        yo = (k == 1 ? 0 : (k == 2 ? 1 : 4))   # SH block offset in a 9-vec
        for c in 1:C
            wl = iw[c, k]
            base = _foff(o1, o2, o3, k) + (c - 1) * dk
            for m in 1:dk
                V[base + m, e] = wl * Y[yo + m, e] * ue
            end
        end
    end
    for c in 1:C
        V[_foff(o1, o2, o3, 1) + c, e] += ib0[c] * ue
    end
end

# Per edge: dense g = env_W·x + env_b (length 3C) and w = tp_W·x + tp_b (length nw).
@kernel inbounds=true function allegro_dense_kernel!(out, @Const(W), @Const(b), @Const(X), rows, H)
    e = @index(Global, Linear)
    for r in 1:rows
        acc = b[r]
        for h in 1:H
            acc += W[r, h] * X[h, e]
        end
        out[r, e] = acc
    end
end

# Per centre atom: Env_i = (1/avg) Σ_{e∈edges(i)} g[l,c]·Y[l,·]  (density trick, gather form).
@kernel inbounds=true function allegro_env_kernel!(Env, @Const(G), @Const(Y), @Const(off),
                                                   C, o1, o2, o3, avg, fd)
    i = @index(Global, Linear)
    T = eltype(Env)
    for f in 1:fd
        Env[f, i] = zero(T)
    end
    e0 = off[i]; e1 = off[i + 1]
    for e in (e0 + 1):e1
        for k in 1:3
            dk = _fdim(k)
            yo = (k == 1 ? 0 : (k == 2 ? 1 : 4))
            for c in 1:C
                gcl = G[(k - 1) * C + c, e]
                base = _foff(o1, o2, o3, k) + (c - 1) * dk
                for m in 1:dk
                    Env[base + m, i] += gcl * Y[yo + m, e]
                end
            end
        end
    end
    inv = one(T) / avg
    for f in 1:fd
        Env[f, i] *= inv
    end
end

# Per edge: P = TP_uvu(V_e, Env_{centre(e)}, w_e).
@kernel inbounds=true function allegro_tp_kernel!(P, @Const(V), @Const(Env), @Const(W),
                                                  @Const(ecenter), @Const(p_k1), @Const(p_k2),
                                                  @Const(p_k3), @Const(p_woff), @Const(poff),
                                                  @Const(cg_m1), @Const(cg_m2), @Const(cg_m3),
                                                  @Const(cg_val), np, C, fd, o1, o2, o3)
    e = @index(Global, Linear)
    T = eltype(P)
    for f in 1:fd
        P[f, e] = zero(T)
    end
    ic = ecenter[e]
    for p in 1:np
        k1 = p_k1[p]; k2 = p_k2[p]; k3 = p_k3[p]
        woff = p_woff[p]
        t0 = poff[p]; t1 = poff[p + 1]
        for c in 1:C
            wc = W[woff + c, e]
            for t in (t0 + 1):t1
                m1 = cg_m1[t]; m2 = cg_m2[t]; m3 = cg_m3[t]
                v = cg_val[t]
                xi = V[_fidx(o1, o2, o3, k1, c, m1), e]
                yi = Env[_fidx(o1, o2, o3, k2, c, m2), ic]
                P[_fidx(o1, o2, o3, k3, c, m3), e] += wc * v * xi * yi
            end
        end
    end
end

# Per edge: scalar resnet x ← x + silu(x_W·[x; P0e] + x_b)·u.
@kernel inbounds=true function allegro_resnet_kernel!(Xout, @Const(Xin), @Const(P), @Const(xW),
                                                      @Const(xb), @Const(u), H, C, o1)
    e = @index(Global, Linear)
    T = eltype(Xout)
    ue = u[e]
    for h in 1:H
        acc = xb[h]
        for k in 1:H
            acc += xW[h, k] * Xin[k, e]
        end
        for c in 1:C
            acc += xW[h, H + c] * P[o1 + c, e]   # 0e scalars of P are the first C entries
        end
        s = acc / (one(T) + exp(-acc))
        Xout[h, e] = Xin[h, e] + s * ue
    end
end

# Per edge: equivariant linear V ← lin(P). lin_w packed as (C, 3C): block k uses cols (k-1)C+1:kC.
@kernel inbounds=true function allegro_eqlin_kernel!(Vout, @Const(P), @Const(lw), @Const(lb0),
                                                     C, o1, o2, o3)
    e = @index(Global, Linear)
    for k in 1:3
        dk = _fdim(k)
        base = _foff(o1, o2, o3, k)
        for m in 1:dk
            for co in 1:C
                acc = (k == 1) ? lb0[co] : zero(eltype(Vout))
                for ci in 1:C
                    acc += lw[co, (k - 1) * C + ci] * P[base + (ci - 1) * dk + m, e]
                end
                Vout[base + (co - 1) * dk + m, e] = acc
            end
        end
    end
end

# Per edge: readout E_e = out_W·x + out_b.
@kernel inbounds=true function allegro_readout_kernel!(Eedge, @Const(X), @Const(oW), @Const(ob), H)
    e = @index(Global, Linear)
    acc = ob[1]
    for h in 1:H
        acc += oW[1, h] * X[h, e]
    end
    Eedge[e] = acc
end

# ---- host orchestration ----------------------------------------------------------------------

# Upload a host array to `backend` with element type T (ints kept as Int32).
_dev(backend, A::AbstractArray{<:Integer}) = (d = KernelAbstractions.allocate(backend, Int32, size(A)); copyto!(d, Int32.(A)); d)
_devf(backend, ::Type{T}, A::AbstractArray) where {T} = (d = KernelAbstractions.allocate(backend, T, size(A)); copyto!(d, T.(A)); d)

"""
    build_allegro_gpu(m::AllegroModel, backend, ::Type{T}) -> AllegroGPU

Move an [`AllegroModel`](@ref)'s weights and CG/path tables onto `backend` with element type `T`
(`Float32` for Metal, `Float64` for CUDA/CPU). Build once and reuse across systems.
"""
function build_allegro_gpu(m::AllegroModel, backend, ::Type{T}) where {T}
    C = m.C; fd = m.feat.dim
    o1 = m.feat.offsets[1]; o2 = m.feat.offsets[2]; o3 = m.feat.offsets[3]
    np = length(m.paths.k); nw = m.paths.n_weights
    p_k1 = Int32[m.paths.k[p][1] for p in 1:np]
    p_k2 = Int32[m.paths.k[p][2] for p in 1:np]
    p_k3 = Int32[m.paths.k[p][3] for p in 1:np]
    p_woff = Int32[m.paths.weight_offset[p] for p in 1:np]
    poff = Int32.(m.cg.poff)
    MT = typeof(_devf(backend, T, m.emb_W1)); VT = typeof(_devf(backend, T, m.emb_b1))
    IT = typeof(_dev(backend, p_k1)); VTI = typeof(_devf(backend, T, m.cg.val))
    env_W = MT[]; env_b = VT[]; tp_W = MT[]; tp_b = VT[]; x_W = MT[]; x_b = VT[]
    lin_w = MT[]; lin_b0 = VT[]
    for lw in m.layers
        push!(env_W, _devf(backend, T, lw.env_W)); push!(env_b, _devf(backend, T, lw.env_b))
        push!(tp_W,  _devf(backend, T, lw.tp_W));  push!(tp_b,  _devf(backend, T, lw.tp_b))
        push!(x_W,   _devf(backend, T, lw.x_W));   push!(x_b,   _devf(backend, T, lw.x_b))
        # pack lin weights (C,C) per block into (C, 3C)
        packed = zeros(T, C, 3C)
        for k in 1:3
            packed[:, (k-1)*C+1:k*C] .= T.(lw.lin.weights[k])
        end
        push!(lin_w, _devf(backend, T, packed)); push!(lin_b0, _devf(backend, T, lw.lin.biases[1]))
    end
    return AllegroGPU{T, VT, MT, IT, VTI}(
        C, m.H, m.nb, m.S, m.L, m.env_p, fd, nw, np, o1, o2, o3, T(m.r_c), T(m.avg_nn),
        _devf(backend, T, m.emb_W1), _devf(backend, T, m.emb_b1),
        _devf(backend, T, m.emb_W2), _devf(backend, T, m.emb_b2),
        _devf(backend, T, m.init_w), _devf(backend, T, m.init_b0),
        _devf(backend, T, m.out_W),  _devf(backend, T, m.out_b),
        env_W, env_b, tp_W, tp_b, x_W, x_b, lin_w, lin_b0,
        _dev(backend, p_k1), _dev(backend, p_k2), _dev(backend, p_k3), _dev(backend, p_woff),
        _dev(backend, poff), _dev(backend, m.cg.m1), _dev(backend, m.cg.m2), _dev(backend, m.cg.m3),
        _devf(backend, T, m.cg.val))
end

"""
    compute_allegro_energy_ka(m, coords, species, boundary; backend=get_backend(coords),
                              gpu=build_allegro_gpu(m, backend, T), workgroup=64) -> T

GPU-portable total Allegro energy via KernelAbstractions. `coords` may live on any KA backend
(CPU, CUDA, Metal). The neighbour list is built on the host (O(N²)); the forward runs on `backend`.
Pass a prebuilt `gpu` (device model) to avoid re-uploading the weights each call.
"""
function compute_allegro_energy_ka(m::AllegroModel, coords::AbstractVector{<:SVector{3}},
                                   species::AbstractVector{<:Integer}, boundary;
                                   backend = KernelAbstractions.get_backend(coords),
                                   T::Type = eltype(eltype(coords)),
                                   gpu::AllegroGPU = build_allegro_gpu(m, backend, T),
                                   workgroup::Int = 64)
    ch = Array(coords)
    n = length(ch)
    nbr = neighbour_lists(ch, boundary, eltype(eltype(ch))(m.r_c))
    off = Vector{Int32}(undef, n + 1); off[1] = 0
    ecenter = Int32[]; ej = Int32[]; rhx = T[]; rhy = T[]; rhz = T[]; dd = T[]
    for i in 1:n
        for (j, d, rhat) in nbr[i]
            push!(ecenter, i); push!(ej, j)
            push!(rhx, T(rhat[1])); push!(rhy, T(rhat[2])); push!(rhz, T(rhat[3])); push!(dd, T(d))
        end
        off[i + 1] = Int32(length(ej))
    end
    ne = length(ej)
    C = gpu.C; H = gpu.H; nb = gpu.nb; S = gpu.S; fd = gpu.fd; nw = gpu.nw
    o1 = gpu.o1; o2 = gpu.o2; o3 = gpu.o3
    if ne == 0
        return zero(T)
    end
    spi = _dev(backend, Int32[species[ecenter[e]] for e in 1:ne])
    spj = _dev(backend, Int32[species[ej[e]] for e in 1:ne])
    off_d = _dev(backend, off); ec_d = _dev(backend, ecenter)
    rhx_d = _devf(backend, T, rhx); rhy_d = _devf(backend, T, rhy); rhz_d = _devf(backend, T, rhz)
    d_d = _devf(backend, T, dd)
    z2 = (a, b) -> KernelAbstractions.zeros(backend, T, a, b)
    Y = z2(9, ne); u = KernelAbstractions.zeros(backend, T, ne); R = z2(nb, ne)
    X = z2(H, ne); A1 = z2(H, ne); V = z2(fd, ne); G = z2(3C, ne)
    Env = z2(fd, n); W = z2(nw, ne); P = z2(fd, ne); Xn = z2(H, ne); Vn = z2(fd, ne)
    Eedge = KernelAbstractions.zeros(backend, T, ne)

    allegro_geom_kernel!(backend, workgroup)(Y, u, R, rhx_d, rhy_d, rhz_d, d_d, gpu.r_c, gpu.env_p, nb; ndrange=ne)
    allegro_emb_kernel!(backend, workgroup)(X, A1, R, spi, spj, gpu.emb_W1, gpu.emb_b1, gpu.emb_W2, gpu.emb_b2, nb, S, H; ndrange=ne)
    allegro_init_kernel!(backend, workgroup)(V, Y, u, gpu.init_w, gpu.init_b0, C, o1, o2, o3; ndrange=ne)
    KernelAbstractions.synchronize(backend)

    for l in 1:gpu.L
        allegro_dense_kernel!(backend, workgroup)(G, gpu.env_W[l], gpu.env_b[l], X, 3C, H; ndrange=ne)
        KernelAbstractions.synchronize(backend)
        allegro_env_kernel!(backend, workgroup)(Env, G, Y, off_d, C, o1, o2, o3, gpu.avg_nn, fd; ndrange=n)
        allegro_dense_kernel!(backend, workgroup)(W, gpu.tp_W[l], gpu.tp_b[l], X, nw, H; ndrange=ne)
        KernelAbstractions.synchronize(backend)
        allegro_tp_kernel!(backend, workgroup)(P, V, Env, W, ec_d, gpu.p_k1, gpu.p_k2, gpu.p_k3, gpu.p_woff, gpu.poff, gpu.cg_m1, gpu.cg_m2, gpu.cg_m3, gpu.cg_val, gpu.np, C, fd, o1, o2, o3; ndrange=ne)
        KernelAbstractions.synchronize(backend)
        allegro_resnet_kernel!(backend, workgroup)(Xn, X, P, gpu.x_W[l], gpu.x_b[l], u, H, C, o1; ndrange=ne)
        allegro_eqlin_kernel!(backend, workgroup)(Vn, P, gpu.lin_w[l], gpu.lin_b0[l], C, o1, o2, o3; ndrange=ne)
        KernelAbstractions.synchronize(backend)
        X, Xn = Xn, X
        V, Vn = Vn, V
    end

    allegro_readout_kernel!(backend, workgroup)(Eedge, X, gpu.out_W, gpu.out_b, H; ndrange=ne)
    KernelAbstractions.synchronize(backend)
    return T(sum(Eedge))
end
