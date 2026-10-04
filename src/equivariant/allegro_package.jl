# Bit-exact native port of the real `nequip-allegro` (allegro 0.8.3) Allegro model forward. Unlike the
# simplified `AllegroModel` in allegro_model.jl, this reproduces the package's *exact* operations, so
# loading the package's own exported weights reproduces its energy and forces to ~1e-6. It is the
# non-circular external validation of the native implementation (see test/allegro_package_reference.py
# for the reference generator and the loader `load_allegro_package` in ext/MollyHDF5Ext.jl).
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
@inline _pkg_cut(x, p) = x < 1 ?
    1 - ((p+1)*(p+2)/2)*x^p + p*(p+2)*x^(p+1) - (p*(p+1)/2)*x^(p+2) : zero(x)
@inline _irrep_of(i) = i == 1 ? 1 : (i <= 4 ? 2 : 3)   # which l-block (1,2,3) a SH component belongs to

"""
    allegro_package_total_energy(m::AllegroPackageModel, coords, species) -> T

Total energy of the bit-exact Allegro port. `coords` are `SVector{3}` in Å; `species` are **0-based**
type indices into `m.type_names` (the package convention). Reproduces the `nequip-allegro` package
energy to numerical precision when `m` is loaded from the package's exported weights.
"""
function allegro_package_total_energy(m::AllegroPackageModel{Tw},
        coords::AbstractVector{<:SVector{3}}, species::AbstractVector{<:Integer}) where Tw
    T = promote_type(Tw, eltype(eltype(coords)))
    n = length(coords); S = m.S; C = m.C; rc = m.r_max
    edges = Tuple{Int,Int}[]
    for i in 1:n, j in 1:n
        i == j && continue
        @inbounds norm(coords[j] - coords[i]) < rc && push!(edges, (i, j))
    end
    ne = length(edges)
    ne == 0 && return zero(T)
    ec = Int[e[1] for e in edges]
    SH = Matrix{T}(undef, ne, 9); embS = Matrix{T}(undef, ne, S)
    @inbounds for (e, (i, j)) in enumerate(edges)
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
    allegro_package_forces(m::AllegroPackageModel, coords, species) -> Vector{SVector{3}}

Analytic forces `F = -∂E/∂r` of the bit-exact Allegro port, by reverse... forward-mode automatic
differentiation (`ForwardDiff`) of [`allegro_package_total_energy`](@ref). `coords` are `SVector{3}`
in Å and `species` are 0-based type indices; the forces are in eV/Å and reproduce the package's own
autograd forces to numerical precision.
"""
function allegro_package_forces(m::AllegroPackageModel, coords::AbstractVector{<:SVector{3}},
                                species::AbstractVector{<:Integer})
    n = length(coords)
    x0 = Vector{Float64}(undef, 3n)
    @inbounds for i in 1:n, k in 1:3
        x0[3 * (i - 1) + k] = coords[i][k]
    end
    E(x) = allegro_package_total_energy(m,
        [SVector{3,eltype(x)}(x[3i - 2], x[3i - 1], x[3i]) for i in 1:n], species)
    g = ForwardDiff.gradient(E, x0)
    return [SVector{3,Float64}(-g[3i - 2], -g[3i - 1], -g[3i]) for i in 1:n]
end

"""
    load_allegro_package(path; T=Float64) -> AllegroPackageModel

Load the real `nequip-allegro` package weights (exported by `test/allegro_package_reference.py` to
`allegro_package_weights.h5`) into a bit-exact native model. Requires `HDF5` to be loaded; the method
is defined in `ext/MollyHDF5Ext.jl`.
"""
function load_allegro_package end
