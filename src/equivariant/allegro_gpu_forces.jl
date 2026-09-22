# GPU-portable Allegro analytic FORCES (KernelAbstractions). This is the reverse pass that mirrors
# the hand-written CPU backward in `allegro_energy_and_forces` (allegro_model.jl), but written as a
# sequence of data-parallel kernels over the same flat directed-edge layout as the energy forward
# (allegro_gpu.jl), so the same code runs on the KA CPU backend, CUDA and Metal.
#
# The forward is re-run with a per-layer tape (X/V layer inputs, tp weights W, tp output P and the
# pooled environment Env, one device array per layer). The backward then walks the layers in reverse:
#   readout → per-layer {eqlinear, scalar resnet, tensor product, environment pooling} → two-body /
#   initial-latent chain → Cartesian gradient.
# Edge-local adjoints (Pbar, Vbar, Wbar, Gbar, Da) live in global scratch columns (one per edge) so
# no kernel needs a dynamically sized local buffer — the one thing that would break portability. The
# environment adjoint is accumulated with a per-atom GATHER over each atom's own edge range (like the
# forward env kernel, no atomics); only the final scatter of ∂E/∂r into the per-atom force array uses
# `Atomix.@atomic` (global float atomics, which CUDA and Metal both support). No intermediate
# synchronize: kernels on one backend run in submission order, each seeing the previous one's writes.

using KernelAbstractions: @kernel, @index, @Const

# ---- backward kernels ------------------------------------------------------------------------

# ∂E/∂x_final = out_W row (same for every edge); ∂E/∂V_final = 0 (readout is scalar-only).
@kernel inbounds=true function allegro_xbar_init_kernel!(Xbar, @Const(oW), H)
    e = @index(Global, Linear)
    for h in 1:H
        Xbar[h, e] = oW[1, h]
    end
end

# Per edge: eqlinear VJP (Vbar→Pbar) + scalar-resnet VJP. Writes Pbar (equivariant-linear part),
# the resnet pre-activation adjoint Da, the identity part of XinBar, and accumulates u-adjoint Ubar.
# `a` (resnet pre-activation) is recomputed from the taped layer input X and forward tp output P.
@kernel inbounds=true function allegro_bwd_eqlin_resnet_kernel!(Pbar, Da, Ubar, XinBar,
        @Const(Xbar), @Const(Vbar), @Const(Xin), @Const(Pfwd), @Const(lw), @Const(xW), @Const(xb),
        @Const(u), fd, H, C, o1, o2, o3)
    e = @index(Global, Linear)
    T = eltype(Pbar)
    ue = u[e]
    # Pbar = eqlinear_vjp(lin, Vbar):  Pbar[k,ci,m] += Σ_co lw[co,(k-1)C+ci]·Vbar[k,co,m]
    for f in 1:fd
        Pbar[f, e] = zero(T)
    end
    for k in 1:3
        dk = _fdim(k)
        base = _foff(o1, o2, o3, k)
        for ci in 1:C
            for m in 1:dk
                acc = zero(T)
                for co in 1:C
                    acc += lw[co, (k - 1) * C + ci] * Vbar[base + (co - 1) * dk + m, e]
                end
                Pbar[base + (ci - 1) * dk + m, e] += acc
            end
        end
    end
    # resnet: x_out = x_in + silu(a)·u, a = xW·[x_in; P0e] + xb
    ub_acc = zero(T)
    for h in 1:H
        a = xb[h]
        for k in 1:H
            a += xW[h, k] * Xin[k, e]
        end
        for c in 1:C
            a += xW[h, H + c] * Pfwd[o1 + c, e]
        end
        s = one(T) / (one(T) + exp(-a))
        sa = a * s
        xb_e = Xbar[h, e]
        ub_acc += xb_e * sa
        Da[h, e] = xb_e * (s * (one(T) + a * (one(T) - s))) * ue   # xbar·silu'(a)·u
        XinBar[h, e] = xb_e                                        # identity residual
    end
    Ubar[e] += ub_acc
end

# Per edge: finish the resnet backward matmul — XinBar += xW[:,1:H]ᵀ·Da; Pbar[0e] += xW[:,H+c]ᵀ·Da.
@kernel inbounds=true function allegro_bwd_resnet_matmul_kernel!(XinBar, Pbar, @Const(Da),
        @Const(xW), H, C, o1)
    e = @index(Global, Linear)
    T = eltype(XinBar)
    for k in 1:H
        acc = zero(T)
        for h in 1:H
            acc += xW[h, k] * Da[h, e]
        end
        XinBar[k, e] += acc
    end
    for c in 1:C
        acc = zero(T)
        for h in 1:H
            acc += xW[h, H + c] * Da[h, e]
        end
        Pbar[o1 + c, e] += acc
    end
end

# Per edge: tensor-product VJP for the per-edge outputs — V-adjoint (VinBar) and weight adjoint
# (Wbar). Mirrors allegro_tp_kernel! exactly: P[k3,c,m3] += wc·v·xi·yi with xi=V, yi=Env, wc=W.
@kernel inbounds=true function allegro_bwd_tp_vw_kernel!(VinBar, Wbar, @Const(Pbar), @Const(V),
        @Const(Env), @Const(W), @Const(ecenter), @Const(p_k1), @Const(p_k2), @Const(p_k3),
        @Const(p_woff), @Const(poff), @Const(cg_m1), @Const(cg_m2), @Const(cg_m3), @Const(cg_val),
        np, nw, C, fd, o1, o2, o3)
    e = @index(Global, Linear)
    T = eltype(VinBar)
    for f in 1:fd
        VinBar[f, e] = zero(T)
    end
    for r in 1:nw
        Wbar[r, e] = zero(T)
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
                i1 = _fidx(o1, o2, o3, k1, c, m1)
                i2 = _fidx(o1, o2, o3, k2, c, m2)
                i3 = _fidx(o1, o2, o3, k3, c, m3)
                xi = V[i1, e]; yi = Env[i2, ic]; pb = Pbar[i3, e]
                VinBar[i1, e] += wc * v * yi * pb
                Wbar[woff + c, e] += v * xi * yi * pb
            end
        end
    end
end

# Per edge: XinBar += tp_Wᵀ·Wbar (weight adjoint feeds the layer-input scalar latent).
@kernel inbounds=true function allegro_bwd_tp_xin_kernel!(XinBar, @Const(Wbar), @Const(tpW), H, nw)
    e = @index(Global, Linear)
    T = eltype(XinBar)
    for h in 1:H
        acc = zero(T)
        for r in 1:nw
            acc += tpW[r, h] * Wbar[r, e]
        end
        XinBar[h, e] += acc
    end
end

# Per CENTRE atom: Env-adjoint gather — Envbar[i] = Σ_{e∈edges(i)} (∂P_e/∂Env_i)ᵀ·Pbar_e. Each atom
# owns a contiguous edge range, so this needs no atomics (mirrors the forward env kernel).
@kernel inbounds=true function allegro_bwd_env_gather_kernel!(Envbar, @Const(Pbar), @Const(V),
        @Const(W), @Const(off), @Const(p_k1), @Const(p_k2), @Const(p_k3), @Const(p_woff),
        @Const(poff), @Const(cg_m1), @Const(cg_m2), @Const(cg_m3), @Const(cg_val),
        np, C, fd, o1, o2, o3)
    i = @index(Global, Linear)
    T = eltype(Envbar)
    for f in 1:fd
        Envbar[f, i] = zero(T)
    end
    e0 = off[i]; e1 = off[i + 1]
    for e in (e0 + 1):e1
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
                    pb = Pbar[_fidx(o1, o2, o3, k3, c, m3), e]
                    Envbar[_fidx(o1, o2, o3, k2, c, m2), i] += wc * v * xi * pb
                end
            end
        end
    end
end

# Per edge: environment-pooling VJP part 1 — the density-trick backward. Recomputes the forward
# per-(l,channel) weight g, scatters Envbar (of the centre atom) to the SH adjoint Ybar and to the
# g-adjoint Gbar. Env_i = (1/avg) Σ_e g_e[l,c]·Y_e ⇒ Ybar += (Envbar/avg)·g, Gbar = Σ_m (Envbar/avg)·Y.
@kernel inbounds=true function allegro_bwd_envpool_g_kernel!(Gbar, Ybar, @Const(Envbar),
        @Const(Xin), @Const(Y), @Const(envW), @Const(envb), @Const(ecenter), avg, C, o1, o2, o3, H)
    e = @index(Global, Linear)
    T = eltype(Gbar)
    ic = ecenter[e]
    invavg = one(T) / avg
    for k in 1:3
        dk = _fdim(k)
        yo = (k == 1 ? 0 : (k == 2 ? 1 : 4))
        base = _foff(o1, o2, o3, k)
        for c in 1:C
            row = (k - 1) * C + c
            gcl = envb[row]
            for h in 1:H
                gcl += envW[row, h] * Xin[h, e]
            end
            acc_g = zero(T)
            for m in 1:dk
                eb = Envbar[base + (c - 1) * dk + m, ic] * invavg
                acc_g += eb * Y[yo + m, e]
                Ybar[yo + m, e] += eb * gcl
            end
            Gbar[row, e] = acc_g
        end
    end
end

# Per edge: environment-pooling VJP part 2 — XinBar += env_Wᵀ·Gbar.
@kernel inbounds=true function allegro_bwd_envpool_xin_kernel!(XinBar, @Const(Gbar), @Const(envW),
        H, C)
    e = @index(Global, Linear)
    T = eltype(XinBar)
    for h in 1:H
        acc = zero(T)
        for r in 1:(3 * C)
            acc += envW[r, h] * Gbar[r, e]
        end
        XinBar[h, e] += acc
    end
end

# Per edge: precompute dh1 = (emb_W2ᵀ·xbar0)·silu'(h1) for the two-body embedding MLP backward.
@kernel inbounds=true function allegro_bwd_emb_dh1_kernel!(Dh1, @Const(Xbar), @Const(u),
        @Const(coords), @Const(ecenter), @Const(ej), @Const(species), @Const(W1), @Const(b1),
        @Const(W2), bx, by, bz, rc, env_p, nb, S, H)
    e = @index(Global, Linear)
    T = eltype(Dh1)
    ic = ecenter[e]; jj = ej[e]
    zi = species[ic]; zj = species[jj]
    dx, dy, dz = _edge_vec(coords[ic], coords[jj], bx, by, bz)
    d = sqrt(dx * dx + dy * dy + dz * dz)
    invd = one(T) / d
    ue = u[e]
    pref = sqrt(T(2) / rc); kk = T(pi) / rc
    for k in 1:H
        h1 = b1[k]
        for q in 1:nb
            Rq = pref * sin(q * kk * d) * invd * ue     # R_q = B_q·u
            h1 += W1[k, q] * Rq
        end
        h1 += W1[k, nb + zi] + W1[k, nb + S + zj]
        da1 = zero(T)
        for h in 1:H
            da1 += W2[h, k] * Xbar[h, e]
        end
        s = one(T) / (one(T) + exp(-h1))
        Dh1[k, e] = da1 * (s * (one(T) + h1 * (one(T) - s)))
    end
end

# Per edge: initial-latent + two-body + geometry backward → Cartesian gradient, atomically scattered
# into the per-atom force array F (F = -∂E/∂r). Consumes the accumulated Ybar/Ubar and the layer-0
# adjoints Xbar (x⁰) and Vbar (V⁰); Dh1 carries the embedding-MLP adjoint from the previous kernel.
@kernel inbounds=true function allegro_bwd_twobody_kernel!(F, @Const(Xbar), @Const(Vbar),
        @Const(Ybar), @Const(Ubar), @Const(Dh1), @Const(Y), @Const(u), @Const(coords),
        @Const(ecenter), @Const(ej), @Const(W1), @Const(iw), @Const(ib0),
        bx, by, bz, rc, env_p, nb, S, H, C, o1, o2, o3)
    e = @index(Global, Linear)
    T = eltype(F)
    ic = ecenter[e]; jj = ej[e]
    dx, dy, dz = _edge_vec(coords[ic], coords[jj], bx, by, bz)
    d = sqrt(dx * dx + dy * dy + dz * dz)
    invd = one(T) / d; invd2 = invd * invd
    ue = u[e]
    rhx = dx * invd; rhy = dy * invd; rhz = dz * invd
    # init: V⁰[k,c,m] = iw[c,k]·Y[k,m]·u (+ ib0[c]·u on 0e). Accumulate Ybar (SH) and ub (envelope).
    ub = Ubar[e]
    for k in 1:3
        dk = _fdim(k)
        yo = (k == 1 ? 0 : (k == 2 ? 1 : 4))
        base = _foff(o1, o2, o3, k)
        for c in 1:C
            wl = iw[c, k]
            for m in 1:dk
                vb = Vbar[base + (c - 1) * dk + m, e]
                ub += wl * Y[yo + m, e] * vb
                # NB: Ybar is read below straight from the array; fold the init SH-adjoint into a
                # local so we don't need to write it back (single reader, this same thread).
            end
        end
    end
    for c in 1:C
        ub += ib0[c] * Vbar[o1 + c, e]
    end
    # two-body embedding: R_q = B_q·u, both depend on d. ds_in[q] = Σ_k W1[k,q]·Dh1[k]. dEdd via
    # the Bessel-basis derivative; ub picks up the R = B·u envelope path.
    pref = sqrt(T(2) / rc); kk = T(pi) / rc
    dEdd = zero(T)
    for q in 1:nb
        Bq = pref * sin(q * kk * d) * invd
        dBq = pref * (q * kk * cos(q * kk * d) * invd - sin(q * kk * d) * invd2)
        dsq = zero(T)
        for k in 1:H
            dsq += W1[k, q] * Dh1[k, e]
        end
        ub += dsq * Bq
        dEdd += dsq * dBq * ue
    end
    _, du = poly_envelope_grad(d, rc, env_p)
    dEdd += ub * du
    # geometry: Y depends on r̂ (SH Jacobian J[q,b] = ∂Y_q/∂r_b); u,R depend on d = |r|. Fold in the
    # init SH-adjoint (iw·u·Vbar) alongside the accumulated Ybar.
    r_raw = SVector{3,T}(dx, dy, dz)
    _, J = _real_sph_harm_grad2(r_raw)
    gx = dEdd * rhx; gy = dEdd * rhy; gz = dEdd * rhz
    for k in 1:3
        dk = _fdim(k)
        yo = (k == 1 ? 0 : (k == 2 ? 1 : 4))
        base = _foff(o1, o2, o3, k)
        for m in 1:dk
            yb = Ybar[yo + m, e]
            for c in 1:C
                yb += iw[c, k] * ue * Vbar[base + (c - 1) * dk + m, e]
            end
            q = yo + m
            gx += J[q, 1] * yb; gy += J[q, 2] * yb; gz += J[q, 3] * yb
        end
    end
    # r = coords[j] - coords[i]: dEdc[j] += dEdr, dEdc[i] -= dEdr; F = -dEdc.
    Atomix.@atomic F[1, jj] += -gx
    Atomix.@atomic F[2, jj] += -gy
    Atomix.@atomic F[3, jj] += -gz
    Atomix.@atomic F[1, ic] += gx
    Atomix.@atomic F[2, ic] += gy
    Atomix.@atomic F[3, ic] += gz
end

# ---- host orchestration ----------------------------------------------------------------------

"""
    compute_allegro_forces_ka(m, coords, species, boundary; backend, T, gpu, workgroup)
        -> (E::T, F)

GPU-portable total Allegro energy `E` and analytic forces `F = -∂E/∂r` via KernelAbstractions —
the device counterpart of [`allegro_energy_and_forces`](@ref). `coords` may live on any KA backend
(CPU, CUDA, Metal); the neighbour list, geometry, the taped forward and the whole reverse pass run
on `backend`. `F` is returned as a `(3, n)` array on `backend` (column `i` is the force on atom `i`);
`Array(F)` brings it to the host. Pass a prebuilt `gpu` (device model) to avoid re-uploading weights.
"""
function compute_allegro_forces_ka(m::AllegroModel, coords::AbstractVector{<:SVector{3}},
                                   species::AbstractVector{<:Integer}, boundary;
                                   backend = KernelAbstractions.get_backend(coords),
                                   T::Type = eltype(eltype(coords)),
                                   gpu::AllegroGPU = build_allegro_gpu(m, backend, T),
                                   workgroup::Int = 64)
    n = length(coords)
    C = gpu.C; H = gpu.H; nb = gpu.nb; S = gpu.S; fd = gpu.fd; nw = gpu.nw; L = gpu.L
    o1 = gpu.o1; o2 = gpu.o2; o3 = gpu.o3
    rc2 = gpu.r_c^2
    bx, by, bz = if boundary === nothing
        (zero(T), zero(T), zero(T))
    else
        sl = boundary.side_lengths
        (T(ustrip(sl[1])), T(ustrip(sl[2])), T(ustrip(sl[3])))
    end
    cdev = coords
    # ---- device neighbour build: count → host prefix-sum → fill (as in the energy path) ----
    counts = KernelAbstractions.zeros(backend, Int32, n)
    allegro_count_kernel!(backend, workgroup)(counts, cdev, n, T(rc2), bx, by, bz; ndrange=n)
    KernelAbstractions.synchronize(backend)
    counts_h = Array(counts)
    off_h = Vector{Int32}(undef, n + 1); off_h[1] = 0
    @inbounds for i in 1:n
        off_h[i + 1] = off_h[i] + counts_h[i]
    end
    ne = Int(off_h[n + 1])
    F = KernelAbstractions.zeros(backend, T, 3, n)
    ne == 0 && return (zero(T), F)
    off_d = _dev(backend, off_h)
    ecenter = KernelAbstractions.allocate(backend, Int32, ne)
    ej = KernelAbstractions.allocate(backend, Int32, ne)
    allegro_fill_kernel!(backend, workgroup)(ecenter, ej, cdev, off_d, n, T(rc2), bx, by, bz; ndrange=n)
    species_d = _dev(backend, Int32.(collect(species)))

    z2 = (a, b) -> KernelAbstractions.zeros(backend, T, a, b)
    z1 = a -> KernelAbstractions.zeros(backend, T, a)
    Y = z2(9, ne); u = z1(ne); R = z2(nb, ne); A1 = z2(H, ne)
    # per-layer tape (layer input X/V, tp weights W, tp output P, pooled Env); L+1 X/V slots.
    Xt = [z2(H, ne)  for _ in 1:(L + 1)]
    Vt = [z2(fd, ne) for _ in 1:(L + 1)]
    Wt = [z2(nw, ne) for _ in 1:L]
    Pt = [z2(fd, ne) for _ in 1:L]
    Et = [z2(fd, n)  for _ in 1:L]
    G  = z2(3C, ne)

    # ---- taped forward ----
    allegro_geom_kernel!(backend, workgroup)(Y, u, R, cdev, ecenter, ej, bx, by, bz, gpu.r_c, gpu.env_p, nb; ndrange=ne)
    allegro_emb_kernel!(backend, workgroup)(Xt[1], A1, R, species_d, ecenter, ej, gpu.emb_W1, gpu.emb_b1, gpu.emb_W2, gpu.emb_b2, nb, S, H; ndrange=ne)
    allegro_init_kernel!(backend, workgroup)(Vt[1], Y, u, gpu.init_w, gpu.init_b0, C, o1, o2, o3; ndrange=ne)
    for l in 1:L
        allegro_dense_kernel!(backend, workgroup)(G, gpu.env_W[l], gpu.env_b[l], Xt[l], 3C, H; ndrange=ne)
        allegro_env_kernel!(backend, workgroup)(Et[l], G, Y, off_d, C, o1, o2, o3, gpu.avg_nn, fd; ndrange=n)
        allegro_dense_kernel!(backend, workgroup)(Wt[l], gpu.tp_W[l], gpu.tp_b[l], Xt[l], nw, H; ndrange=ne)
        allegro_tp_kernel!(backend, workgroup)(Pt[l], Vt[l], Et[l], Wt[l], ecenter, gpu.p_k1, gpu.p_k2, gpu.p_k3, gpu.p_woff, gpu.poff, gpu.cg_m1, gpu.cg_m2, gpu.cg_m3, gpu.cg_val, gpu.np, C, fd, o1, o2, o3; ndrange=ne)
        allegro_resnet_kernel!(backend, workgroup)(Xt[l + 1], Xt[l], Pt[l], gpu.x_W[l], gpu.x_b[l], u, H, C, o1; ndrange=ne)
        allegro_eqlin_kernel!(backend, workgroup)(Vt[l + 1], Pt[l], gpu.lin_w[l], gpu.lin_b0[l], C, o1, o2, o3; ndrange=ne)
    end
    Eedge = z1(ne)
    allegro_readout_kernel!(backend, workgroup)(Eedge, Xt[L + 1], gpu.out_W, gpu.out_b, H; ndrange=ne)

    # ---- backward ----
    Xbar = z2(H, ne); Vbar = z2(fd, ne)              # adjoints of the current layer's outputs
    XinBar = z2(H, ne); VinBar = z2(fd, ne)          # adjoints of the current layer's inputs
    Pbar = z2(fd, ne); Da = z2(H, ne); Gbar = z2(3C, ne); Wbar = z2(nw, ne)
    Envbar = z2(fd, n)
    Ybar = z2(9, ne); Ubar = z1(ne); Dh1 = z2(H, ne) # accumulated across all layers, zero-init

    allegro_xbar_init_kernel!(backend, workgroup)(Xbar, gpu.out_W, H; ndrange=ne)   # Vbar stays 0
    for l in L:-1:1
        allegro_bwd_eqlin_resnet_kernel!(backend, workgroup)(Pbar, Da, Ubar, XinBar, Xbar, Vbar, Xt[l], Pt[l], gpu.lin_w[l], gpu.x_W[l], gpu.x_b[l], u, fd, H, C, o1, o2, o3; ndrange=ne)
        allegro_bwd_resnet_matmul_kernel!(backend, workgroup)(XinBar, Pbar, Da, gpu.x_W[l], H, C, o1; ndrange=ne)
        allegro_bwd_tp_vw_kernel!(backend, workgroup)(VinBar, Wbar, Pbar, Vt[l], Et[l], Wt[l], ecenter, gpu.p_k1, gpu.p_k2, gpu.p_k3, gpu.p_woff, gpu.poff, gpu.cg_m1, gpu.cg_m2, gpu.cg_m3, gpu.cg_val, gpu.np, nw, C, fd, o1, o2, o3; ndrange=ne)
        allegro_bwd_tp_xin_kernel!(backend, workgroup)(XinBar, Wbar, gpu.tp_W[l], H, nw; ndrange=ne)
        allegro_bwd_env_gather_kernel!(backend, workgroup)(Envbar, Pbar, Vt[l], Wt[l], off_d, gpu.p_k1, gpu.p_k2, gpu.p_k3, gpu.p_woff, gpu.poff, gpu.cg_m1, gpu.cg_m2, gpu.cg_m3, gpu.cg_val, gpu.np, C, fd, o1, o2, o3; ndrange=n)
        allegro_bwd_envpool_g_kernel!(backend, workgroup)(Gbar, Ybar, Envbar, Xt[l], Y, gpu.env_W[l], gpu.env_b[l], ecenter, gpu.avg_nn, C, o1, o2, o3, H; ndrange=ne)
        allegro_bwd_envpool_xin_kernel!(backend, workgroup)(XinBar, Gbar, gpu.env_W[l], H, C; ndrange=ne)
        Xbar, XinBar = XinBar, Xbar
        Vbar, VinBar = VinBar, Vbar
    end
    allegro_bwd_emb_dh1_kernel!(backend, workgroup)(Dh1, Xbar, u, cdev, ecenter, ej, species_d, gpu.emb_W1, gpu.emb_b1, gpu.emb_W2, bx, by, bz, gpu.r_c, gpu.env_p, nb, S, H; ndrange=ne)
    allegro_bwd_twobody_kernel!(backend, workgroup)(F, Xbar, Vbar, Ybar, Ubar, Dh1, Y, u, cdev, ecenter, ej, gpu.emb_W1, gpu.init_w, gpu.init_b0, bx, by, bz, gpu.r_c, gpu.env_p, nb, S, H, C, o1, o2, o3; ndrange=ne)
    return (T(sum(Eedge)), F)
end
