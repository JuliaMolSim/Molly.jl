# Extension loading a native Allegro potential's weights from an HDF5 file. Triggered by
# `using HDF5`. Everything else (the equivariant primitives, the model forward, the analytic
# forces and the AtomsCalculators wiring) lives in core Molly; only reading the HDF5 file needs
# this extension.

module MollyHDF5Ext

using Molly
using Molly: build_allegro_model
using HDF5

# HDF5.jl reads a Python (row-major) array with its dimensions reversed relative to numpy; this
# restores the numpy orientation so `W * x` uses W as (out, in).
reverse_dims(A) = ndims(A) <= 1 ? A : permutedims(A, reverse(ntuple(identity, ndims(A))))

"""
    AllegroPotential(path::AbstractString; T=Float32)

Load a native Allegro equivariant potential from an HDF5 file exported by
`test/allegro_reference.py`. See the core docstring for usage notes.
"""
function Molly.AllegroPotential(path::AbstractString; T::Type=Float32)
    h5open(path, "r") do f
        cfg = attrs(f["config"])
        C = Int(cfg["C"])
        H = Int(cfg["H"])
        nb = Int(cfg["nb"])
        S = Int(cfg["S"])
        L = Int(cfg["L"])
        env_p = Int(cfg["env_p"])
        rc = Float64(cfg["rc"])
        rd(p) = reverse_dims(read(f[p]))
        layers = map(0:L-1) do li
            g = "layer$li"
            (tp_W=rd("$g/tp_W"), tp_b=rd("$g/tp_b"), x_W=rd("$g/x_W"), x_b=rd("$g/x_b"),
             lin_w=rd("$g/lin_w"), lin_b0=rd("$g/lin_b0"))
        end
        weights = (emb_W1=rd("emb_W1"), emb_b1=rd("emb_b1"), emb_W2=rd("emb_W2"), emb_b2=rd("emb_b2"),
                   init_w=rd("init_w"), init_b0=rd("init_b0"), out_W=rd("out_W"), out_b=rd("out_b"),
                   layers=layers)
        model = build_allegro_model(; C=C, H=H, nb=nb, S=S, L=L, env_p=env_p, r_c=rc, weights=weights, T=T)
        # species map: the HDF5 file may carry a "species" list of element symbols.
        species_map = if haskey(f, "species")
            Dict{String,Int}(string(s) => i for (i, s) in enumerate(read(f["species"])))
        else
            Dict{String,Int}("__$(i)__" => i for i in 1:S)
        end
        return Molly.AllegroPotential(model, species_map, T(rc), Ref{Any}(nothing))
    end
end

end # module
