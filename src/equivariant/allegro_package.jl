# Bit-exact native port of the real `nequip-allegro` (allegro 0.8.3) Allegro model forward: loading
# the package's own exported weights reproduces its energy and forces to ~1e-6. This is the native
# implementation behind `AllegroPotential`, validated non-circularly against the actual package (see
# test/allegro_package_reference.py for the reference generator and the loader `load_allegro_package`
# in ext/MollyHDF5Ext.jl). Energy is computed here; forces are `allegro_package_forces` (AD).
#
# Op sequence (per directed edge i<-j, r = r_j - r_i, d = |r|, r̂ = r/d), config l_max=2:
#   normed = d / r_max;  u = DimeNet polynomial cutoff (p);  bessel_k = sinc(normed·k)·k·u  (k=1..nb)
#   two-body embed: cat(center_embed[Z_i], neighbor_embed[Z_j]) ⊙ (bessel · W_basis)        -> 32 (= S)
#   scalar_embed_mlp: silu-MLP(32 -> 2·? -> 32)
#   SH: e3nn component-normalised real spherical harmonics (9 comps, l=0,1,2)
#   tensor features x^0[u,i] = SH[i] · env_weight[u, irrep(i)]        (u=1..C tensor channels)
#   Allegro layers (num_layers): env-weight SH -> scatter to centre atom -> ×1/√avg_nn ->
#     strided Wigner-3j tensor product (stored w3j already × √(2ℓ+1)) -> extract 0e scalars ->
#     DenseNet latent silu-MLP; scalar features accumulate across layers.
#   edge readout: silu-MLP(C·(L+1) -> 1);  atom energy = (Σ_edges E_e)/√(2·avg_nn)  (per centre atom)

struct AllegroPackageModel{T}
    S::Int; C::Int; nb::Int; L::Int; p::Int
    r_max::T; avg_nn::T
    type_names::Vector{String}
    bessel_w::Vector{T}
    center_embed::Matrix{T}      # (S÷2, ntypes)
    neighbor_embed::Matrix{T}    # (S÷2, ntypes)
    basis_W::Matrix{T}           # (S, nb)
    semb_W0::Matrix{T}; semb_W2::Matrix{T}
    env_W::Matrix{T}             # tensor_embed env linear (3C, S)
    proj_W::Matrix{T}            # first-layer projection (S+3C, S)
    lat_W0::Vector{Matrix{T}}; lat_W2::Vector{Matrix{T}}  # per-layer latent MLP
    tp_w::Vector{Matrix{T}}      # per layer (C, n_paths)
    tp_w3j::Vector{Array{T,4}}   # per layer (n_paths, 9, 9, nk); a diagonal layer is expanded to full
    tp_nk::Vector{Int}
    ro_W0::Matrix{T}; ro_W2::Matrix{T}
end

@inline _silu(x) = x / (1 + exp(-x))
@inline function _silu_grad(x)
    s = 1 / (1 + exp(-x))
    return s * (1 + x * (1 - s))
end
@inline _pkg_cut(x, p) = x < 1 ?
    1 - ((p+1)*(p+2)/2)*x^p + p*(p+2)*x^(p+1) - (p*(p+1)/2)*x^(p+2) : zero(x)
@inline _pkg_cut_grad(x, p) = x < 1 ?
    -((p+1)*(p+2)/2)*p*x^(p-1) + p*(p+2)*(p+1)*x^p - (p*(p+1)/2)*(p+2)*x^(p+1) : zero(x)
@inline _irrep_of(i) = i == 1 ? 1 : (i <= 4 ? 2 : 3)   # which l-block (1,2,3) a SH component belongs to

# Directed edges (centre i, neighbour j) within the cutoff, open box. Callers in an MD loop can build
# this once and pass it to the `edges` kwarg to skip the rebuild on every energy/forces evaluation.
function pkg_edges_cpu(coords, rc)
    n = length(coords); edg = Tuple{Int,Int}[]
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        norm(coords[j] - coords[i]) < rc && push!(edg, (i, j))
    end
    return edg
end

"""
    allegro_package_total_energy(m::AllegroPackageModel, coords, species) -> T

Total energy of the bit-exact Allegro port. `coords` are `SVector{3}` in Å; `species` are **0-based**
type indices into `m.type_names` (the package convention). Reproduces the `nequip-allegro` package
energy to numerical precision when `m` is loaded from the package's exported weights.
"""
function allegro_package_total_energy(m::AllegroPackageModel{Tw},
        coords::AbstractVector{<:SVector{3}}, species::AbstractVector{<:Integer};
        edges = nothing) where Tw
    T = promote_type(Tw, eltype(eltype(coords)))
    n = length(coords); S = m.S; C = m.C; rc = m.r_max
    edg = edges === nothing ? pkg_edges_cpu(coords, rc) : edges
    ne = length(edg)
    ne == 0 && return zero(T)
    ec = Int[e[1] for e in edg]
    SH = Matrix{T}(undef, ne, 9); embS = Matrix{T}(undef, ne, S)
    @inbounds for (e, (i, j)) in enumerate(edg)
        r = coords[j] - coords[i]; d = norm(r)
        SH[e, :] = collect(real_sph_harm(2, r / d))
        x = d / rc; u = _pkg_cut(x, m.p)
        bessel = T[sinc(x * m.bessel_w[k]) * m.bessel_w[k] * u for k in 1:m.nb]
        te = vcat(m.center_embed[:, species[i] + 1], m.neighbor_embed[:, species[j] + 1])
        embS[e, :] = m.semb_W2 * _silu.(m.semb_W0 * (te .* (m.basis_W * bessel)))
    end
    # initial tensor features x^0[e,u,i] and layer-0 scalar features + env weights
    tf = zeros(T, ne, C, 9)
    acc = [Matrix{T}(undef, ne, S) for _ in 1:(m.L + 1)]
    env_w = Matrix{T}(undef, ne, 3C)
    @inbounds for e in 1:ne
        w = m.env_W * embS[e, :]
        for u in 1:C, i in 1:9
            tf[e, u, i] = SH[e, i] * w[(u - 1) * 3 + _irrep_of(i)]
        end
        pr = m.proj_W * embS[e, :]
        acc[1][e, :] = pr[1:S]; env_w[e, :] = pr[S + 1:S + 3C]
    end
    for l in 1:m.L
        # env-weighted SH scattered to centre atoms, normalised by 1/√avg_nn
        node = zeros(T, n, C, 9)
        @inbounds for e in 1:ne, u in 1:C, i in 1:9
            node[ec[e], u, i] += SH[e, i] * env_w[e, (u - 1) * 3 + _irrep_of(i)]
        end
        node ./= sqrt(m.avg_nn)
        w3j = m.tp_w3j[l]; wtp = m.tp_w[l]; nk = m.tp_nk[l]; npath = size(w3j, 1)
        out = zeros(T, ne, C, nk)
        @inbounds for e in 1:ne, u in 1:C, k in 1:nk
            s = zero(T)
            for i in 1:9, j in 1:9
                wv = zero(T)
                for pth in 1:npath
                    wv += wtp[u, pth] * w3j[pth, i, j, k]
                end
                s += tf[e, u, i] * node[ec[e], u, j] * wv
            end
            out[e, u, k] = s
        end
        tf = out
        # DenseNet latent: input = cat(all accumulated scalar features so far, this TP's 0e scalars).
        # Built into an explicit buffer (no splatting) so the forward stays type-stable for Enzyme.
        inp = Vector{T}(undef, l * S + C)
        @inbounds for e in 1:ne
            for q in 1:l, c in 1:S
                inp[(q - 1) * S + c] = acc[q][e, c]
            end
            for u in 1:C
                inp[l * S + u] = tf[e, u, 1]
            end
            lat = m.lat_W2[l] * _silu.(m.lat_W0[l] * inp)
            acc[l + 1][e, :] = lat[1:S]
            if l < m.L
                env_w[e, :] = lat[S + 1:S + 3C]
            end
        end
    end
    # readout + per-centre-atom edgewise reduce with 1/√(2·avg_nn)
    atom = zeros(T, n)
    ef = Vector{T}(undef, (m.L + 1) * S)
    @inbounds for e in 1:ne
        for q in 1:(m.L + 1), c in 1:S
            ef[(q - 1) * S + c] = acc[q][e, c]
        end
        atom[ec[e]] += (m.ro_W2 * _silu.(m.ro_W0 * ef))[1]
    end
    return sum(atom) / sqrt(2 * m.avg_nn)
end

"""
    allegro_package_energy_and_forces(m::AllegroPackageModel, coords, species) -> (E, F)

Total energy `E` and analytic forces `F = -∂E/∂r` of the bit-exact Allegro port, via a hand-written
reverse pass through every op (readout + edgewise reduce, the DenseNet latents, the strided Wigner-3j
tensor products, the environment scatter, the two-body embedding, and the spherical-harmonic Jacobian).
`coords` are `SVector{3}` in Å, `species` are 0-based type indices; forces are in eV/Å and reproduce
the package's own autograd forces to numerical precision.
"""
function allegro_package_energy_and_forces(m::AllegroPackageModel{Tw},
        coords::AbstractVector{<:SVector{3}}, species::AbstractVector{<:Integer};
        edges = nothing) where Tw
    T = promote_type(Tw, eltype(eltype(coords)))
    n = length(coords); S = m.S; C = m.C; rc = m.r_max; L = m.L; nb = m.nb
    invs = one(T) / sqrt(m.avg_nn)
    edg = edges === nothing ? pkg_edges_cpu(coords, rc) : edges
    ne = length(edg)
    F = [zero(SVector{3,T}) for _ in 1:n]
    ne == 0 && return (zero(T), F)
    ec = Int[e[1] for e in edg]
    Hs = size(m.semb_W0, 1)                              # scalar-embed hidden width

    # ---------- taped forward ----------
    SH = Matrix{T}(undef, ne, 9); dists = Vector{T}(undef, ne)
    rhat = Vector{SVector{3,T}}(undef, ne); rraw = Vector{SVector{3,T}}(undef, ne)
    te = Matrix{T}(undef, ne, S); pre_semb = Matrix{T}(undef, ne, Hs); embS = Matrix{T}(undef, ne, S)
    @inbounds for (e, (i, j)) in enumerate(edg)
        r = coords[j] - coords[i]; d = norm(r); rh = r / d
        dists[e] = d; rhat[e] = rh; rraw[e] = r
        SH[e, :] = collect(real_sph_harm(2, rh))
        x = d / rc; u = _pkg_cut(x, m.p)
        bessel = T[sinc(x * m.bessel_w[k]) * m.bessel_w[k] * u for k in 1:nb]
        tev = vcat(m.center_embed[:, species[i] + 1], m.neighbor_embed[:, species[j] + 1])
        te[e, :] = tev
        pre = m.semb_W0 * (tev .* (m.basis_W * bessel)); pre_semb[e, :] = pre
        embS[e, :] = m.semb_W2 * _silu.(pre)
    end
    w_env = Matrix{T}(undef, ne, 3C)
    tf_hist = Vector{Array{T,3}}(undef, L + 1); tf_hist[1] = zeros(T, ne, C, 9)
    acc = [Matrix{T}(undef, ne, S) for _ in 1:(L + 1)]
    envw = [Matrix{T}(undef, ne, 3C) for _ in 1:L]       # envw[l] = env weights used by layer l
    @inbounds for e in 1:ne
        w = m.env_W * embS[e, :]; w_env[e, :] = w
        for u in 1:C, i in 1:9
            tf_hist[1][e, u, i] = SH[e, i] * w[(u - 1) * 3 + _irrep_of(i)]
        end
        pr = m.proj_W * embS[e, :]; acc[1][e, :] = pr[1:S]; envw[1][e, :] = pr[S + 1:S + 3C]
    end
    node_hist = Vector{Array{T,3}}(undef, L); lat_pre = Vector{Matrix{T}}(undef, L)
    for l in 1:L
        node = zeros(T, n, C, 9)
        @inbounds for e in 1:ne, u in 1:C, i in 1:9
            node[ec[e], u, i] += SH[e, i] * envw[l][e, (u - 1) * 3 + _irrep_of(i)]
        end
        node .*= invs; node_hist[l] = node
        w3j = m.tp_w3j[l]; wtp = m.tp_w[l]; nk = m.tp_nk[l]; npath = size(w3j, 1); tfin = tf_hist[l]
        out = zeros(T, ne, C, nk)
        @inbounds for e in 1:ne, u in 1:C, k in 1:nk
            s = zero(T)
            for i in 1:9, j in 1:9
                wv = zero(T); for p in 1:npath; wv += wtp[u, p] * w3j[p, i, j, k]; end
                s += tfin[e, u, i] * node[ec[e], u, j] * wv
            end
            out[e, u, k] = s
        end
        tf_hist[l + 1] = out
        lp = Matrix{T}(undef, ne, size(m.lat_W0[l], 1))
        @inbounds for e in 1:ne
            inp = Vector{T}(undef, l * S + C)
            for q in 1:l, c in 1:S; inp[(q - 1) * S + c] = acc[q][e, c]; end
            for u in 1:C; inp[l * S + u] = out[e, u, 1]; end
            pre = m.lat_W0[l] * inp; lp[e, :] = pre
            lat = m.lat_W2[l] * _silu.(pre)
            acc[l + 1][e, :] = lat[1:S]
            l < L && (envw[l + 1][e, :] = lat[S + 1:S + 3C])
        end
        lat_pre[l] = lp
    end
    p_ro = Matrix{T}(undef, ne, size(m.ro_W0, 1)); atom = zeros(T, n)
    @inbounds for e in 1:ne
        ef = Vector{T}(undef, (L + 1) * S)
        for q in 1:(L + 1), c in 1:S; ef[(q - 1) * S + c] = acc[q][e, c]; end
        pr = m.ro_W0 * ef; p_ro[e, :] = pr
        atom[ec[e]] += (m.ro_W2 * _silu.(pr))[1]
    end
    E = sum(atom) / sqrt(2 * m.avg_nn)

    # ---------- reverse pass ----------
    scale = one(T) / sqrt(2 * m.avg_nn)
    ro_w2v = vec(m.ro_W2)
    acc_bar = [zeros(T, ne, S) for _ in 1:(L + 1)]
    SH_bar = zeros(T, ne, 9); embS_bar = zeros(T, ne, S)
    @inbounds for e in 1:ne                              # readout + edgewise reduce
        ef_bar = (m.ro_W0' * (ro_w2v .* _silu_grad.(p_ro[e, :]))) .* scale
        for q in 1:(L + 1), c in 1:S; acc_bar[q][e, c] += ef_bar[(q - 1) * S + c]; end
    end
    tf_bar = zeros(T, ne, C, m.tp_nk[L])                 # adjoint of tf_hist[L+1]
    envw_bar = zeros(T, ne, 3C)                          # adjoint of the env weights used by layer `l`
    for l in L:-1:1
        nk = m.tp_nk[l]
        @inbounds for e in 1:ne                          # latent_l backward
            lat_bar = Vector{T}(undef, l < L ? S + 3C : S)
            for c in 1:S; lat_bar[c] = acc_bar[l + 1][e, c]; end
            l < L && (for c in 1:3C; lat_bar[S + c] = envw_bar[e, c]; end)
            inp_bar = m.lat_W0[l]' * (_silu_grad.(lat_pre[l][e, :]) .* (m.lat_W2[l]' * lat_bar))
            for q in 1:l, c in 1:S; acc_bar[q][e, c] += inp_bar[(q - 1) * S + c]; end
            for u in 1:C; tf_bar[e, u, 1] += inp_bar[l * S + u]; end
        end
        w3j = m.tp_w3j[l]; wtp = m.tp_w[l]; npath = size(w3j, 1)
        ww = zeros(T, C, 9, 9, nk)                       # ww[u,i,j,k] = Σ_p wtp[u,p] w3j[p,i,j,k]
        @inbounds for u in 1:C, i in 1:9, j in 1:9, k in 1:nk
            s = zero(T); for p in 1:npath; s += wtp[u, p] * w3j[p, i, j, k]; end; ww[u, i, j, k] = s
        end
        tfin = tf_hist[l]; node = node_hist[l]
        tfin_bar = zeros(T, ne, C, 9); node_bar = zeros(T, n, C, 9)
        @inbounds for e in 1:ne                          # strided TP backward
            c = ec[e]
            for u in 1:C, k in 1:nk
                ob = tf_bar[e, u, k]; ob == 0 && continue
                for i in 1:9, j in 1:9
                    w = ww[u, i, j, k]
                    tfin_bar[e, u, i] += ob * node[c, u, j] * w
                    node_bar[c, u, j] += ob * tfin[e, u, i] * w
                end
            end
        end
        new_envw_bar = zeros(T, ne, 3C)                  # env scatter backward
        @inbounds for e in 1:ne
            c = ec[e]
            for u in 1:C, j in 1:9
                nbv = node_bar[c, u, j] * invs; idx = (u - 1) * 3 + _irrep_of(j)
                new_envw_bar[e, idx] += nbv * SH[e, j]
                SH_bar[e, j] += nbv * envw[l][e, idx]
            end
        end
        tf_bar = tfin_bar; envw_bar = new_envw_bar
    end
    # tensor embed + proj backward → embS_bar, SH_bar
    @inbounds for e in 1:ne
        wenv_bar = zeros(T, 3C)
        for u in 1:C, i in 1:9
            tb = tf_bar[e, u, i]; idx = (u - 1) * 3 + _irrep_of(i)
            SH_bar[e, i] += tb * w_env[e, idx]; wenv_bar[idx] += tb * SH[e, i]
        end
        pr_bar = Vector{T}(undef, S + 3C)
        for c in 1:S; pr_bar[c] = acc_bar[1][e, c]; end
        for c in 1:3C; pr_bar[S + c] = envw_bar[e, c]; end
        embS_bar[e, :] = m.env_W' * wenv_bar .+ m.proj_W' * pr_bar
    end
    # scalar-embed MLP → bessel → d; SH Jacobian → r; scatter to atoms
    @inbounds for e in 1:ne
        (i, j) = edg[e]
        pre_bar = _silu_grad.(pre_semb[e, :]) .* (m.semb_W2' * embS_bar[e, :])
        basis_bar = (m.semb_W0' * pre_bar) .* te[e, :]
        bessel_bar = m.basis_W' * basis_bar
        d = dists[e]; x = d / rc; u = _pkg_cut(x, m.p); du = _pkg_cut_grad(x, m.p) / rc
        dEdd = zero(T)
        for k in 1:nb
            bw = m.bessel_w[k]; arg = x * bw
            dbdd = bw * (cosc(arg) * bw / rc * u + sinc(arg) * du)   # ∂(sinc(x·bw)·bw·u)/∂d
            dEdd += bessel_bar[k] * dbdd
        end
        _, J = real_sph_harm_grad(2, rraw[e])
        rh = rhat[e]; gx = dEdd * rh[1]; gy = dEdd * rh[2]; gz = dEdd * rh[3]
        for q in 1:9
            gx += J[q, 1] * SH_bar[e, q]; gy += J[q, 2] * SH_bar[e, q]; gz += J[q, 3] * SH_bar[e, q]
        end
        dEdr = SVector{3,T}(gx, gy, gz)     # r = coords[j] - coords[i]; F = -∂E/∂coords
        F[j] -= dEdr; F[i] += dEdr
    end
    return (E, F)
end

"""
    allegro_package_forces(m::AllegroPackageModel, coords, species) -> Vector{SVector{3}}

Analytic forces `F = -∂E/∂r` of the bit-exact Allegro port (see
[`allegro_package_energy_and_forces`](@ref)). `coords` are `SVector{3}` in Å, `species` are 0-based
type indices; forces are in eV/Å and reproduce the package's autograd forces to numerical precision.
"""
allegro_package_forces(m::AllegroPackageModel, coords::AbstractVector{<:SVector{3}},
                       species::AbstractVector{<:Integer}) =
    allegro_package_energy_and_forces(m, coords, species)[2]

"""
    load_allegro_package(path; T=Float64) -> AllegroPackageModel

Load the real `nequip-allegro` package weights (exported by `test/allegro_package_reference.py` to
`allegro_package_weights.h5`) into a bit-exact native model. Requires `HDF5` to be loaded; the method
is defined in `ext/MollyHDF5Ext.jl`.
"""
function load_allegro_package end
