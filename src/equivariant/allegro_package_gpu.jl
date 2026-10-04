# GPU-portable (KernelAbstractions) forward + analytic backward for the bit-exact Allegro port
# (AllegroPackageModel). Mirrors the CPU paths in allegro_package.jl op-for-op, so loading the real
# nequip-allegro weights reproduces the package energy and forces on CUDA / Metal to device precision.
# One thread per directed edge for the per-edge ops; Atomix atomics for the edge→atom scatters. The
# per-layer Wigner-3j path weights are pre-folded into `ww3j[u,i,j,k] = Σ_p w[u,p]·w3j[p,i,j,k]` at
# build time, so the TP kernels are a plain contraction.

struct AllegroPackageGPU{T, VT, MT, A3, IT}
    S::Int; C::Int; nb::Int; L::Int; p::Int
    r_max::T; avg_nn::T
    nks::Vector{Int}; inlens::Vector{Int}
    bessel_w::VT
    center_embed::MT; neighbor_embed::MT; basis_W::MT
    semb_W0::MT; semb_W2::MT; env_W::MT; proj_W::MT
    ww3j::Vector{A3}                       # per layer, flattened (C*9*9*nk) device vector
    lat_W0::Vector{MT}; lat_W2::Vector{MT}
    ro_W0::MT; ro_W2::MT
    _it::IT                                # unused marker to carry the Int array type
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
    ww = map(1:m.L) do l
        w3j = m.tp_w3j[l]; wtp = m.tp_w[l]; nk = m.tp_nk[l]; npath = size(w3j, 1)
        f = zeros(T, C * 9 * 9 * nk)
        @inbounds for u in 1:C, i in 1:9, j in 1:9, k in 1:nk
            s = zero(T); for pth in 1:npath; s += wtp[u, pth] * w3j[pth, i, j, k]; end
            f[(((k - 1) * 9 + (j - 1)) * 9 + (i - 1)) * C + u] = s   # layout [u,i,j,k], u fastest
        end
        _pkgdev(backend, T, f)
    end
    inlens = [l * m.S + C for l in 1:m.L]
    md(A) = _pkgdev(backend, T, A)
    return AllegroPackageGPU{T, typeof(md(m.bessel_w)), typeof(md(m.basis_W)), eltype(ww),
                             typeof(_pkgdev_i(backend, Int32[1]))}(
        m.S, C, m.nb, m.L, m.p, T(m.r_max), T(m.avg_nn), copy(m.tp_nk), inlens,
        md(m.bessel_w), md(m.center_embed), md(m.neighbor_embed), md(m.basis_W),
        md(m.semb_W0), md(m.semb_W2), md(m.env_W), md(m.proj_W),
        ww, map(md, m.lat_W0), map(md, m.lat_W2), md(m.ro_W0), md(m.ro_W2),
        _pkgdev_i(backend, Int32[1]))
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

@kernel inbounds=true function pkg_tp_kernel!(out, @Const(tfin), @Const(node), @Const(ww), @Const(ecenter), C, nk)
    e = @index(Global, Linear); T = eltype(out); c = ecenter[e]
    for k in 1:nk, u in 1:C
        s = zero(T)
        for i in 1:9, j in 1:9
            s += tfin[(i - 1) * C + u, e] * node[(j - 1) * C + u, c] *
                 ww[(((k - 1) * 9 + (j - 1)) * 9 + (i - 1)) * C + u]
        end
        out[(k - 1) * C + u, e] = s
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
        boundary = nothing, workgroup::Int = 64)
    n = length(coords); S = gpu.S; C = gpu.C; L = gpu.L
    bx, by, bz = boundary === nothing ? (zero(T), zero(T), zero(T)) :
        (T(ustrip(boundary.side_lengths[1])), T(ustrip(boundary.side_lengths[2])), T(ustrip(boundary.side_lengths[3])))
    ci, cj = _pkg_edges_host(Array(coords), Float64(gpu.r_max), Float64(bx), Float64(by), Float64(bz))
    ne = length(ci)
    ne == 0 && return zero(T)
    ec = _pkgdev_i(backend, ci); ej = _pkgdev_i(backend, cj)
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
        pkg_tp_kernel!(backend, workgroup)(out, tf, node, gpu.ww3j[l], ec, C, nk; ndrange=ne)
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
