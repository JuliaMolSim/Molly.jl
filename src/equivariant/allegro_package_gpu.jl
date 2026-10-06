# GPU-portable (KernelAbstractions) forward + analytic backward for the bit-exact Allegro port
# (AllegroPackageModel). Mirrors the CPU paths in allegro_package.jl op-for-op, so loading the real
# nequip-allegro weights reproduces the package energy and forces on CUDA / Metal to device precision.
# One thread per directed edge for the per-edge ops; Atomix atomics for the edge→atom scatters. The
# per-layer Wigner-3j path weights are pre-folded into `w[u] = Σ_p w[u,p]·w3j[p,i,j,k]` at build time
# and stored as a SPARSE list of the non-zero (i,j,k) output components (angular-momentum selection
# rules zero out ~89%), so the tensor-product kernels iterate only the active paths.

struct AllegroPackageGPU{T, VT, MT, A3, VI}
    S::Int; C::Int; nb::Int; L::Int; p::Int
    r_max::T; avg_nn::T
    nks::Vector{Int}; inlens::Vector{Int}; tpnnz::Vector{Int}
    bessel_w::VT
    center_embed::MT; neighbor_embed::MT; basis_W::MT
    semb_W0::MT; semb_W2::MT; env_W::MT; proj_W::MT
    # Sparse Wigner-3j tensor-product paths per layer: the active (i,j,k) output-component index lists
    # (`tpi`/`tpj`/`tpk`, length `tpnnz`) and the per-(path, channel) folded weights `tpw` (flattened
    # nnz*C, channel fastest). ~89% of (i,j,k) triples are zero by angular-momentum selection rules, so
    # iterating only the active paths makes the TP kernels ~9x cheaper than the dense 9x9xnk loop.
    tpi::Vector{VI}; tpj::Vector{VI}; tpk::Vector{VI}; tpw::Vector{A3}
    lat_W0::Vector{MT}; lat_W2::Vector{MT}
    ro_W0::MT; ro_W2::MT
end

_pkgdev(backend, ::Type{T}, A::AbstractArray) where {T} =
    (d = KernelAbstractions.allocate(backend, T, size(A)); copyto!(d, T.(A)); d)
_pkgdev_i(backend, A::AbstractArray{<:Integer}) =
    (d = KernelAbstractions.allocate(backend, Int32, size(A)); copyto!(d, Int32.(A)); d)

"""
    build_allegro_package_gpu(m::AllegroPackageModel, backend, ::Type{T}) -> AllegroPackageGPU

Move an [`AllegroPackageModel`](@ref)'s weights onto `backend` (`Float32` for Metal, `Float64` for
CUDA/CPU), pre-folding the per-layer path weights into the Wigner-3j tables. Build once and reuse.
"""
function build_allegro_package_gpu(m::AllegroPackageModel, backend, ::Type{T}) where {T}
    C = m.C
    # Fold the per-layer path weights into the Wigner-3j table and keep only the non-zero (i,j,k)
    # output components as a sparse path list (angular-momentum selection rules zero out ~89%).
    sparse = map(1:m.L) do l
        w3j = m.tp_w3j[l]; wtp = m.tp_w[l]; nk = m.tp_nk[l]; npath = size(w3j, 1)
        ti = Int32[]; tj = Int32[]; tk = Int32[]; wv = T[]
        @inbounds for k in 1:nk, j in 1:9, i in 1:9
            vals = ntuple(C) do u
                s = zero(T); for pth in 1:npath; s += wtp[u, pth] * w3j[pth, i, j, k]; end; s
            end
            if any(!iszero, vals)
                push!(ti, i); push!(tj, j); push!(tk, k)
                for u in 1:C; push!(wv, vals[u]); end          # flattened nnz*C, channel u fastest
            end
        end
        (_pkgdev_i(backend, ti), _pkgdev_i(backend, tj), _pkgdev_i(backend, tk), _pkgdev(backend, T, wv), length(ti))
    end
    tpi = [s[1] for s in sparse]; tpj = [s[2] for s in sparse]; tpk = [s[3] for s in sparse]
    tpw = [s[4] for s in sparse]; tpnnz = [s[5] for s in sparse]
    inlens = [l * m.S + C for l in 1:m.L]
    md(A) = _pkgdev(backend, T, A)
    return AllegroPackageGPU{T, typeof(md(m.bessel_w)), typeof(md(m.basis_W)), eltype(tpw), eltype(tpi)}(
        m.S, C, m.nb, m.L, m.p, T(m.r_max), T(m.avg_nn), copy(m.tp_nk), inlens, tpnnz,
        md(m.bessel_w), md(m.center_embed), md(m.neighbor_embed), md(m.basis_W),
        md(m.semb_W0), md(m.semb_W2), md(m.env_W), md(m.proj_W),
        tpi, tpj, tpk, tpw, map(md, m.lat_W0), map(md, m.lat_W2), md(m.ro_W0), md(m.ro_W2))
end

@inline _pkg_ir(i) = i == 1 ? 1 : (i <= 4 ? 2 : 3)
@inline _pkg_silu(x::T) where {T} = x / (one(T) + exp(-x))
@inline function _pkg_silu_grad(x::T) where {T}
    s = one(T) / (one(T) + exp(-x)); return s * (one(T) + x * (one(T) - s))
end
# NB: all coefficients/powers are kept in `T` (no Float64 intermediates) so these compile for Metal.
@inline function _pkg_cutf(x::T, p::Integer) where {T}
    x < one(T) || return zero(T)
    pt = T(p); a = (pt + 1) * (pt + 2) / T(2); b = pt * (pt + 2); c = pt * (pt + 1) / T(2)
    xp = x^p
    return one(T) - a * xp + b * xp * x - c * xp * x * x
end
@inline function _pkg_cutgf(x::T, p::Integer) where {T}
    x < one(T) || return zero(T)
    pt = T(p); a = (pt + 1) * (pt + 2) / T(2); b = pt * (pt + 2); c = pt * (pt + 1) / T(2)
    xm = x^(p - 1)
    return -a * pt * xm + b * (pt + 1) * xm * x - c * (pt + 2) * xm * x * x
end

# edge vector r_j - r_i under an (optional) orthorhombic box (bx,by,bz); minimum image when boxed.
@inline function _pkg_edge(ci, cj, bx::T, by::T, bz::T) where {T}
    dx = cj[1] - ci[1]; dy = cj[2] - ci[2]; dz = cj[3] - ci[3]
    if bx > 0
        dx -= bx * round(dx / bx); dy -= by * round(dy / by); dz -= bz * round(dz / bz)
    end
    return dx, dy, dz
end

# ---- forward kernels (per directed edge) ----
@kernel inbounds=true function pkg_geom_kernel!(SH, bessel, dd, rh, @Const(coords),
        @Const(ecenter), @Const(ej), @Const(bw), bx, by, bz, rc, p, nb)
    e = @index(Global, Linear); T = eltype(SH)
    dx, dy, dz = _pkg_edge(coords[ecenter[e]], coords[ej[e]], T(bx), T(by), T(bz))
    d = sqrt(dx * dx + dy * dy + dz * dz); invd = one(T) / d
    dd[e] = d; rh[1, e] = dx * invd; rh[2, e] = dy * invd; rh[3, e] = dz * invd
    Yv = real_sph_harm(2, SVector{3,T}(dx * invd, dy * invd, dz * invd))
    for q in 1:9; SH[q, e] = Yv[q]; end
    x = d / rc; u = _pkg_cutf(x, p)
    # sinc(x·bw)·bw·u = sin(π·x·bw)/(π·x)·u, written out so it compiles for the GPU (no `sinc` branch)
    for k in 1:nb; bessel[k, e] = sin(T(pi) * x * bw[k]) / (T(pi) * x) * u; end
end

@kernel inbounds=true function pkg_tf0_kernel!(tf0, @Const(SH), @Const(wenv), C)
    e = @index(Global, Linear)
    for i in 1:9
        ir = _pkg_ir(i)
        for u in 1:C
            tf0[(i - 1) * C + u, e] = SH[i, e] * wenv[(u - 1) * 3 + ir, e]
        end
    end
end

@kernel inbounds=true function pkg_envscatter_kernel!(node, @Const(SH), @Const(envw), @Const(ecenter), C)
    e = @index(Global, Linear); c = ecenter[e]
    for j in 1:9
        ir = _pkg_ir(j)
        for u in 1:C
            Atomix.@atomic node[(j - 1) * C + u, c] += SH[j, e] * envw[(u - 1) * 3 + ir, e]
        end
    end
end

# Strided Wigner-3j contraction, sparse over the active (i,j,k) paths. `out` (C*nk, ne) is pre-zeroed
# and accumulated: out[k,u,e] += Σ_paths tfin[i,u,e]·node[j,u,c]·w[path,u].
@kernel inbounds=true function pkg_tp_kernel!(out, @Const(tfin), @Const(node),
        @Const(ti), @Const(tj), @Const(tk), @Const(w), @Const(ecenter), C, nnz)
    e = @index(Global, Linear); c = ecenter[e]
    for t in 1:nnz
        i = ti[t]; j = tj[t]; k = tk[t]; base = (t - 1) * C
        for u in 1:C
            out[(k - 1) * C + u, e] += tfin[(i - 1) * C + u, e] * node[(j - 1) * C + u, c] * w[base + u]
        end
    end
end

@kernel inbounds=true function pkg_readout_scatter_kernel!(atom, @Const(eedge), @Const(ecenter))
    e = @index(Global, Linear)
    Atomix.@atomic atom[ecenter[e]] += eedge[1, e]
end

# build directed edges within the cutoff on the host (open or orthorhombic box) and upload the
# (centre, neighbour) index arrays. Keeps the heavy compute on-device.
function _pkg_edges_host(coords_h, rc, bx, by, bz)
    n = length(coords_h); ci = Int32[]; cj = Int32[]
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        dx = coords_h[j][1] - coords_h[i][1]; dy = coords_h[j][2] - coords_h[i][2]; dz = coords_h[j][3] - coords_h[i][3]
        if bx > 0
            dx -= bx * round(dx / bx); dy -= by * round(dy / by); dz -= bz * round(dz / bz)
        end
        dx * dx + dy * dy + dz * dz < rc^2 && (push!(ci, i); push!(cj, j))
    end
    return ci, cj
end

# Build the directed edge list and upload it to `backend`, returning device (centre, neighbour) index
# vectors ready for the kernels. Callers that already hold a neighbour list (e.g. a System reusing one
# across MD steps) pass it to the `edges` kwarg instead and skip this entirely.
function pkg_build_edges(coords, r_max, bx, by, bz; backend = KernelAbstractions.get_backend(coords))
    ci, cj = _pkg_edges_host(Array(coords), Float64(r_max), Float64(bx), Float64(by), Float64(bz))
    return _pkgdev_i(backend, ci), _pkgdev_i(backend, cj)
end

"""
    compute_allegro_package_energy_ka(m, coords, species; backend, T, gpu, boundary=nothing) -> T

GPU-portable total energy of the bit-exact Allegro port via KernelAbstractions. `coords` may live on
any KA backend; the per-edge MLPs run as device matmuls, the geometry / tensor-product / scatters as
kernels. Pass a prebuilt `gpu` device model to avoid re-uploading weights.
"""
function compute_allegro_package_energy_ka(m::AllegroPackageModel,
        coords::AbstractVector{<:SVector{3}}, species::AbstractVector{<:Integer};
        backend = KernelAbstractions.get_backend(coords),
        T::Type = eltype(eltype(coords)),
        gpu::AllegroPackageGPU = build_allegro_package_gpu(m, backend, T),
        boundary = nothing, workgroup::Int = 64, edges = nothing)
    n = length(coords); S = gpu.S; C = gpu.C; L = gpu.L
    bx, by, bz = boundary === nothing ? (zero(T), zero(T), zero(T)) :
        (T(ustrip(boundary.side_lengths[1])), T(ustrip(boundary.side_lengths[2])), T(ustrip(boundary.side_lengths[3])))
    # Directed edge list (centre ec, neighbour ej) within r_max. Built on-device (O(N) cell list)
    # unless the caller passes a precomputed `edges = (ec, ej)` (e.g. a System's neighbour list reused
    # across MD steps), so the neighbour search is not redone on every evaluation.
    ec, ej = edges === nothing ? pkg_build_edges(coords, gpu.r_max, bx, by, bz; backend=backend) : edges
    ne = length(ec)
    ne == 0 && return zero(T)
    sp = _pkgdev_i(backend, Int32.(collect(species)))
    z2(a, b) = KernelAbstractions.zeros(backend, T, a, b)
    SH = z2(9, ne); bessel = z2(gpu.nb, ne); dd = KernelAbstractions.zeros(backend, T, ne); rh = z2(3, ne)
    pkg_geom_kernel!(backend, workgroup)(SH, bessel, dd, rh, coords, ec, ej, gpu.bessel_w,
        bx, by, bz, gpu.r_max, gpu.p, gpu.nb; ndrange=ne)
    # two-body embedding: type embed ⊙ basis, then scalar MLP (matmuls)
    half = S ÷ 2
    sp_c = sp[ec]; sp_j = sp[ej]
    te = vcat(gpu.center_embed[:, sp_c .+ 1], gpu.neighbor_embed[:, sp_j .+ 1])   # (S, ne)
    tb = te .* (gpu.basis_W * bessel)
    embS = gpu.semb_W2 * _pkg_silu.(gpu.semb_W0 * tb)                              # (S, ne)
    wenv = gpu.env_W * embS                                                        # (3C, ne)
    tf = z2(C * 9, ne)
    pkg_tf0_kernel!(backend, workgroup)(tf, SH, wenv, C; ndrange=ne)
    pr = gpu.proj_W * embS                                                         # (S+3C, ne)
    acc = KernelAbstractions.zeros(backend, T, S, ne, L + 1)
    acc[:, :, 1] = pr[1:S, :]
    envw = pr[S + 1:S + 3C, :]
    invs = one(T) / sqrt(gpu.avg_nn)
    for l in 1:L
        nk = gpu.nks[l]
        node = KernelAbstractions.zeros(backend, T, C * 9, n)
        pkg_envscatter_kernel!(backend, workgroup)(node, SH, envw, ec, C; ndrange=ne)
        node = node .* invs
        out = z2(C * nk, ne)
        pkg_tp_kernel!(backend, workgroup)(out, tf, node, gpu.tpi[l], gpu.tpj[l], gpu.tpk[l], gpu.tpw[l], ec, C, gpu.tpnnz[l]; ndrange=ne)
        inp = vcat(reshape(permutedims(acc[:, :, 1:l], (1, 3, 2)), l * S, ne), out[1:C, :])
        lat = gpu.lat_W2[l] * _pkg_silu.(gpu.lat_W0[l] * inp)                      # (outlen, ne)
        acc[:, :, l + 1] = lat[1:S, :]
        l < L && (envw = lat[S + 1:S + 3C, :])
        tf = out
    end
    ef = reshape(permutedims(acc, (1, 3, 2)), (L + 1) * S, ne)                     # ((L+1)S, ne)
    eedge = gpu.ro_W2 * _pkg_silu.(gpu.ro_W0 * ef)                                 # (1, ne)
    atom = KernelAbstractions.zeros(backend, T, n)
    pkg_readout_scatter_kernel!(backend, workgroup)(atom, eedge, ec; ndrange=ne)
    KernelAbstractions.synchronize(backend)
    return T(sum(atom)) / sqrt(2 * gpu.avg_nn)
end

# GPU-safe sinc and its derivative (Julia's sinc/cosc don't compile for Metal).
@inline _sincg(z::T) where {T} = sin(T(pi) * z) / (T(pi) * z)
@inline _coscg(z::T) where {T} = cos(T(pi) * z) / z - sin(T(pi) * z) / (T(pi) * z * z)

# ---- backward kernels (per directed edge) ----
# Reverse of the sparse TP contraction. tfin_bar and node_bar are pre-zeroed; each active path adds to
# the tf-input gradient (per-edge, no atomic) and the node gradient (scattered to the centre, atomic).
@kernel inbounds=true function pkg_tp_bwd_kernel!(tfin_bar, node_bar, @Const(out_bar),
        @Const(tfin), @Const(node), @Const(ti), @Const(tj), @Const(tk), @Const(w), @Const(ecenter), C, nnz)
    e = @index(Global, Linear); c = ecenter[e]
    for t in 1:nnz
        i = ti[t]; j = tj[t]; k = tk[t]; base = (t - 1) * C
        for u in 1:C
            ob = out_bar[(k - 1) * C + u, e] * w[base + u]
            tfin_bar[(i - 1) * C + u, e] += ob * node[(j - 1) * C + u, c]
            Atomix.@atomic node_bar[(j - 1) * C + u, c] += ob * tfin[(i - 1) * C + u, e]
        end
    end
end

@kernel inbounds=true function pkg_envscatter_bwd_kernel!(envw_bar, SH_bar, @Const(node_bar),
        @Const(envw), @Const(SH), @Const(ecenter), C, invs)
    e = @index(Global, Linear); T = eltype(envw_bar); c = ecenter[e]
    for j in 1:9
        ir = _pkg_ir(j); sb = zero(T)
        for u in 1:C
            nbv = node_bar[(j - 1) * C + u, c] * invs
            envw_bar[(u - 1) * 3 + ir, e] += nbv * SH[j, e]
            sb += nbv * envw[(u - 1) * 3 + ir, e]
        end
        SH_bar[j, e] += sb
    end
end

@kernel inbounds=true function pkg_tf0_bwd_kernel!(wenv_bar, SH_bar, @Const(tf_bar), @Const(wenv), @Const(SH), C)
    e = @index(Global, Linear); T = eltype(wenv_bar)
    for i in 1:9
        ir = _pkg_ir(i); sb = zero(T)
        for u in 1:C
            tb = tf_bar[(i - 1) * C + u, e]
            wenv_bar[(u - 1) * 3 + ir, e] += tb * SH[i, e]
            sb += tb * wenv[(u - 1) * 3 + ir, e]
        end
        SH_bar[i, e] += sb
    end
end

@kernel inbounds=true function pkg_geom_bwd_kernel!(F, @Const(bessel_bar), @Const(SH_bar), @Const(dd),
        @Const(rh), @Const(coords), @Const(ecenter), @Const(ej), @Const(bw), bx, by, bz, rc, p, nb)
    e = @index(Global, Linear); T = eltype(F)
    d = dd[e]; x = d / rc; u = _pkg_cutf(x, p); du = _pkg_cutgf(x, p) / rc
    dEdd = zero(T)
    for k in 1:nb
        bwk = bw[k]; arg = x * bwk
        dbdd = bwk * (_coscg(arg) * bwk / rc * u + _sincg(arg) * du)     # ∂(sinc(x·bw)·bw·u)/∂d
        dEdd += bessel_bar[k, e] * dbdd
    end
    dx, dy, dz = _pkg_edge(coords[ecenter[e]], coords[ej[e]], T(bx), T(by), T(bz))
    _, J = _real_sph_harm_grad2(SVector{3,T}(dx, dy, dz))
    gx = dEdd * rh[1, e]; gy = dEdd * rh[2, e]; gz = dEdd * rh[3, e]
    for q in 1:9
        gx += J[q, 1] * SH_bar[q, e]; gy += J[q, 2] * SH_bar[q, e]; gz += J[q, 3] * SH_bar[q, e]
    end
    i = ecenter[e]; j = ej[e]                   # r = coords[j]-coords[i]; F = -∂E/∂coords
    Atomix.@atomic F[1, j] += -gx; Atomix.@atomic F[2, j] += -gy; Atomix.@atomic F[3, j] += -gz
    Atomix.@atomic F[1, i] += gx;  Atomix.@atomic F[2, i] += gy;  Atomix.@atomic F[3, i] += gz
end

"""
    compute_allegro_package_energy_and_forces_ka(m, coords, species; backend, T, gpu, boundary) -> (E, F)

GPU-portable energy and analytic forces of the bit-exact Allegro port. `F` is a `(3, n)` array on
`backend` (`F = -∂E/∂r`, eV/Å). Mirrors the CPU reverse pass; MLP adjoints are transposed matmuls,
the tensor-product / scatter / geometry adjoints are kernels.
"""
function compute_allegro_package_energy_and_forces_ka(m::AllegroPackageModel,
        coords::AbstractVector{<:SVector{3}}, species::AbstractVector{<:Integer};
        backend = KernelAbstractions.get_backend(coords),
        T::Type = eltype(eltype(coords)),
        gpu::AllegroPackageGPU = build_allegro_package_gpu(m, backend, T),
        boundary = nothing, workgroup::Int = 64, edges = nothing)
    n = length(coords); S = gpu.S; C = gpu.C; L = gpu.L
    bx, by, bz = boundary === nothing ? (zero(T), zero(T), zero(T)) :
        (T(ustrip(boundary.side_lengths[1])), T(ustrip(boundary.side_lengths[2])), T(ustrip(boundary.side_lengths[3])))
    ec, ej = edges === nothing ? pkg_build_edges(coords, gpu.r_max, bx, by, bz; backend=backend) : edges
    ne = length(ec)
    F = KernelAbstractions.zeros(backend, T, 3, n)
    ne == 0 && return (zero(T), F)
    sp = _pkgdev_i(backend, Int32.(collect(species)))
    z2(a, b) = KernelAbstractions.zeros(backend, T, a, b)
    invs = one(T) / sqrt(gpu.avg_nn); scale = one(T) / sqrt(2 * gpu.avg_nn)
    # ---- taped forward ----
    SH = z2(9, ne); bessel = z2(gpu.nb, ne); dd = KernelAbstractions.zeros(backend, T, ne); rh = z2(3, ne)
    pkg_geom_kernel!(backend, workgroup)(SH, bessel, dd, rh, coords, ec, ej, gpu.bessel_w,
        bx, by, bz, gpu.r_max, gpu.p, gpu.nb; ndrange=ne)
    te = vcat(gpu.center_embed[:, sp[ec] .+ 1], gpu.neighbor_embed[:, sp[ej] .+ 1])
    tb = te .* (gpu.basis_W * bessel)
    pre_semb = gpu.semb_W0 * tb
    embS = gpu.semb_W2 * _pkg_silu.(pre_semb)
    wenv = gpu.env_W * embS
    tf = z2(C * 9, ne); pkg_tf0_kernel!(backend, workgroup)(tf, SH, wenv, C; ndrange=ne)
    pr = gpu.proj_W * embS
    acc = KernelAbstractions.zeros(backend, T, S, ne, L + 1); acc[:, :, 1] = pr[1:S, :]
    envw_hist = Vector{Any}(undef, L); envw_hist[1] = pr[S + 1:S + 3C, :]
    tf_hist = Vector{Any}(undef, L + 1); tf_hist[1] = tf
    node_hist = Vector{Any}(undef, L); lat_pre = Vector{Any}(undef, L)
    for l in 1:L
        nk = gpu.nks[l]
        node = KernelAbstractions.zeros(backend, T, C * 9, n)
        pkg_envscatter_kernel!(backend, workgroup)(node, SH, envw_hist[l], ec, C; ndrange=ne)
        node = node .* invs; node_hist[l] = node
        out = z2(C * nk, ne)
        pkg_tp_kernel!(backend, workgroup)(out, tf_hist[l], node, gpu.tpi[l], gpu.tpj[l], gpu.tpk[l], gpu.tpw[l], ec, C, gpu.tpnnz[l]; ndrange=ne)
        tf_hist[l + 1] = out
        inp = vcat(reshape(permutedims(acc[:, :, 1:l], (1, 3, 2)), l * S, ne), out[1:C, :])
        pre = gpu.lat_W0[l] * inp; lat_pre[l] = pre
        lat = gpu.lat_W2[l] * _pkg_silu.(pre)
        acc[:, :, l + 1] = lat[1:S, :]
        l < L && (envw_hist[l + 1] = lat[S + 1:S + 3C, :])
    end
    ef = reshape(permutedims(acc, (1, 3, 2)), (L + 1) * S, ne)
    p_ro = gpu.ro_W0 * ef
    eedge = gpu.ro_W2 * _pkg_silu.(p_ro)
    atom = KernelAbstractions.zeros(backend, T, n)
    pkg_readout_scatter_kernel!(backend, workgroup)(atom, eedge, ec; ndrange=ne)
    # ---- reverse pass ----
    ef_bar = (transpose(gpu.ro_W0) * (transpose(gpu.ro_W2) .* _pkg_silu_grad.(p_ro))) .* scale
    acc_bar = permutedims(reshape(ef_bar, S, L + 1, ne), (1, 3, 2))        # (S, ne, L+1)
    SH_bar = z2(9, ne)
    tf_bar = z2(C * gpu.nks[L], ne); envw_bar = z2(3C, ne)
    for l in L:-1:1
        nk = gpu.nks[l]
        lat_bar = l < L ? vcat(acc_bar[:, :, l + 1], envw_bar) : acc_bar[:, :, l + 1]
        inp_bar = transpose(gpu.lat_W0[l]) * (_pkg_silu_grad.(lat_pre[l]) .* (transpose(gpu.lat_W2[l]) * lat_bar))
        acc_bar[:, :, 1:l] = acc_bar[:, :, 1:l] .+ permutedims(reshape(inp_bar[1:l * S, :], S, l, ne), (1, 3, 2))
        tf_bar[1:C, :] = tf_bar[1:C, :] .+ inp_bar[l * S + 1:l * S + C, :]
        tfin_bar = z2(C * 9, ne); node_bar = KernelAbstractions.zeros(backend, T, C * 9, n)
        pkg_tp_bwd_kernel!(backend, workgroup)(tfin_bar, node_bar, tf_bar, tf_hist[l], node_hist[l],
            gpu.tpi[l], gpu.tpj[l], gpu.tpk[l], gpu.tpw[l], ec, C, gpu.tpnnz[l]; ndrange=ne)
        new_envw_bar = z2(3C, ne)
        pkg_envscatter_bwd_kernel!(backend, workgroup)(new_envw_bar, SH_bar, node_bar, envw_hist[l], SH, ec, C, invs; ndrange=ne)
        tf_bar = tfin_bar; envw_bar = new_envw_bar
    end
    wenv_bar = z2(3C, ne)
    pkg_tf0_bwd_kernel!(backend, workgroup)(wenv_bar, SH_bar, tf_bar, wenv, SH, C; ndrange=ne)
    pr_bar = vcat(acc_bar[:, :, 1], envw_bar)
    embS_bar = transpose(gpu.env_W) * wenv_bar .+ transpose(gpu.proj_W) * pr_bar
    pre_bar = _pkg_silu_grad.(pre_semb) .* (transpose(gpu.semb_W2) * embS_bar)
    basis_bar = (transpose(gpu.semb_W0) * pre_bar) .* te
    bessel_bar = transpose(gpu.basis_W) * basis_bar
    pkg_geom_bwd_kernel!(backend, workgroup)(F, bessel_bar, SH_bar, dd, rh, coords, ec, ej,
        gpu.bessel_w, bx, by, bz, gpu.r_max, gpu.p, gpu.nb; ndrange=ne)
    KernelAbstractions.synchronize(backend)
    return (T(sum(atom)) / sqrt(2 * gpu.avg_nn), F)
end
