# Extension loading a native Allegro potential's weights from an HDF5 file. Triggered by
# `using HDF5`. Everything else (the equivariant primitives, the model forward, the analytic
# forces and the AtomsCalculators wiring) lives in core Molly; only reading the HDF5 file needs
# this extension.

module MollyHDF5Ext

using Molly
using HDF5

# HDF5.jl reads a Python (row-major) array with its dimensions reversed relative to numpy; this
# restores the numpy orientation so `W * x` uses W as (out, in).
reverse_dims(A) = ndims(A) <= 1 ? A : permutedims(A, reverse(ntuple(identity, ndims(A))))

"""
    AllegroPotential(path::AbstractString; T=Float64)

Load the bit-exact native Allegro potential from an HDF5 file of `nequip-allegro` weights exported by
`test/allegro_package_reference.py`. See the core docstring for usage notes.
"""
function Molly.AllegroPotential(path::AbstractString; T::Type=Float64)
    model = Molly.load_allegro_package(path; T=T)
    # element → 0-based type index, the convention the package model uses
    species_map = Dict{String,Int}(name => i - 1 for (i, name) in enumerate(model.type_names))
    return Molly.AllegroPotential(model, species_map, model.r_max)
end

"""
    load_allegro_package(path; T=Float64) -> AllegroPackageModel

Load the real `nequip-allegro` package weights (HDF5 from `test/allegro_package_reference.py`) into a
bit-exact native [`AllegroPackageModel`](@ref). Linear weights are read as `(out, in)` (the h5 reversal
of the package's `(in, out)`); the per-layer `tp` weights and `w3j` tables are restored to the package
orientation `(channel, path)` and `(path, i, j, k)`.
"""
function Molly.load_allegro_package(path::AbstractString; T::Type=Float64)
    h5open(path, "r") do f
        cfg = attrs(f["config"])
        S = Int(cfg["num_scalar_features"]); C = Int(cfg["num_tensor_features"])
        nb = Int(cfg["num_bessels"]); L = Int(cfg["num_layers"]); p = Int(cfg["polynomial_cutoff_p"])
        rmax = T(cfg["r_max"]); avg = T(cfg["avg_num_neighbors"])
        tn = String.(read(f["type_names"]))
        pre = "model__func__"
        g(k) = T.(read(f["w/" * pre * k]))                 # h5 reversal -> (out, in) for linear weights
        full_reverse(A) = permutedims(A, reverse(ntuple(identity, ndims(A))))
        latW0 = Matrix{T}[]; latW2 = Matrix{T}[]
        tpw = Matrix{T}[]; tpw3j = Array{T,4}[]; tpnk = Int[]
        for l in 0:(L - 1)
            push!(latW0, g("allegro__latents__$(l)__mlp__0__weight"))
            push!(latW2, g("allegro__latents__$(l)__mlp__2__weight"))
            push!(tpw, permutedims(g("allegro__tps__$(l)__weights"), (2, 1)))   # -> (C, n_paths)
            w3j = full_reverse(g("allegro__tps__$(l)__w3j"))                     # -> (n_paths, i, [j,] k)
            if ndims(w3j) == 3     # diagonal layer (i==j): expand (p, i, k) -> (p, i, j, k) with j==i
                np_, ni, nk_ = size(w3j)
                w4 = zeros(T, np_, ni, ni, nk_)
                for pth in 1:np_, i in 1:ni, k in 1:nk_
                    w4[pth, i, i, k] = w3j[pth, i, k]
                end
                w3j = w4
            end
            push!(tpw3j, w3j); push!(tpnk, size(w3j, 4))
        end
        return Molly.AllegroPackageModel{T}(S, C, nb, L, p, rmax, avg, tn,
            vec(g("radial_chemical_embed__bessel_encode__bessel_weights")),
            g("radial_chemical_embed__type_embed__center_embed__weight"),
            g("radial_chemical_embed__type_embed__neighbor_embed__weight"),
            g("radial_chemical_embed__type_embed__basis_linear__mlp__0__weight"),
            g("scalar_embed_mlp__mlp_module__mlp__0__weight"),
            g("scalar_embed_mlp__mlp_module__mlp__2__weight"),
            g("tensor_embed__env_embed_linear__mlp__0__weight"),
            g("allegro__first_layer_env_embed_projection__mlp__0__weight"),
            latW0, latW2, tpw, tpw3j, tpnk,
            g("edge_readout__mlp_module__mlp__0__weight"),
            g("edge_readout__mlp_module__mlp__2__weight"))
    end
end

end # module
