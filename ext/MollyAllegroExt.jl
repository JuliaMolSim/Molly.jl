# Extension implementing the native Allegro equivariant potential's model loading and
# AtomsCalculators wiring. Triggered by `using Lux, HDF5` (same weakdeps as MollyLuxExt). The
# equivariant primitives and the model forward live in core Molly (src/equivariant/); this file
# only loads trained weights from HDF5 and connects the model to a Molly `System`.

module MollyAllegroExt

using Molly
using Molly: AllegroModel, build_allegro_model, vector, SVector
import AtomsCalculators
using HDF5
using Unitful
using LinearAlgebra

# HDF5.jl reads a Python (row-major) array with its dimensions reversed relative to numpy; this
# restores the numpy orientation so `W*x` uses W as (out, in).
_revperm(A) = ndims(A) <= 1 ? A : permutedims(A, reverse(ntuple(identity, ndims(A))))

"""
    AllegroPotential(path::AbstractString; T=Float32)

Load a native Allegro equivariant potential from an HDF5 file exported by
`test/allegro_reference.py`. See the core docstring for usage notes.
"""
function Molly.AllegroPotential(path::AbstractString; T::Type=Float32)
    h5open(path, "r") do f
        cfg = attrs(f["config"])
        C = Int(cfg["C"]); H = Int(cfg["H"]); nb = Int(cfg["nb"]); S = Int(cfg["S"])
        L = Int(cfg["L"]); env_p = Int(cfg["env_p"]); rc = Float64(cfg["rc"])
        rd(p) = _revperm(read(f[p]))
        layers = map(0:L-1) do li
            g = "layer$li"
            (tp_W=rd("$g/tp_W"), tp_b=rd("$g/tp_b"), x_W=rd("$g/x_W"), x_b=rd("$g/x_b"),
             lin_w=rd("$g/lin_w"), lin_b0=rd("$g/lin_b0"))
        end
        weights = (emb_W1=rd("emb_W1"), emb_b1=rd("emb_b1"), emb_W2=rd("emb_W2"), emb_b2=rd("emb_b2"),
                   init_w=rd("init_w"), init_b0=rd("init_b0"), out_W=rd("out_W"), out_b=rd("out_b"),
                   layers=layers)
        model = build_allegro_model(; C=C, H=H, nb=nb, S=S, L=L, env_p=env_p, r_c=rc, weights=weights, T=T)
        # species map: HDF5 may carry a "species" list of element symbols; else default to 1..S.
        species_map = if haskey(f, "species")
            Dict{String,Int}(string(s) => i for (i, s) in enumerate(read(f["species"])))
        else
            Dict{String,Int}("__$(i)__" => i for i in 1:S)
        end
        return Molly.AllegroPotential(model, species_map, T(rc), Ref{Any}(nothing))
    end
end

Molly.allegro_data_dir() = error("no artifact configured yet; load AllegroPotential from a local HDF5 path")

# Coordinates: Molly stores unitless coords as nm; the model uses Å (×10). Unitful coords are
# stripped to nm first.
function _coords_to_angstrom(coords)
    c1 = first(coords)
    if eltype(c1) <: Real
        return [SVector{3,Float64}(Float64(c[1]) * 10, Float64(c[2]) * 10, Float64(c[3]) * 10) for c in coords]
    else
        return [SVector{3,Float64}(ustrip(u"nm", c[1]) * 10, ustrip(u"nm", c[2]) * 10, ustrip(u"nm", c[3]) * 10)
                for c in coords]
    end
end

# Scale a boundary from nm to Å (eltype-preserving), so minimum-image distances are in Å.
function _boundary_to_angstrom(b::CubicBoundary)
    sl = b.side_lengths
    vals = eltype(sl) <: Real ? (Float64.(sl) .* 10) : (ustrip.(u"nm", sl) .* 10)
    return CubicBoundary(vals...)
end
_boundary_to_angstrom(::Nothing) = nothing

# Energy in the model's native (eV) units converted to the system's energy units.
function _allegro_energy_to_units(E, energy_units)
    if energy_units == Unitful.NoUnits
        return E                                   # unitless system: return the model number
    elseif dimension(energy_units) == dimension(u"kJ*mol^-1")
        return uconvert(energy_units, E * u"eV" * Unitful.Na)
    else
        return uconvert(energy_units, E * u"eV")
    end
end

function _allegro_force_to_units(f, force_units)
    if force_units == Unitful.NoUnits
        return f
    elseif dimension(force_units) == dimension(u"kJ*mol^-1*nm^-1")
        return uconvert.(force_units, f .* (u"eV/nm") .* Unitful.Na)
    else
        return uconvert.(force_units, f .* u"eV/nm")
    end
end

_species_vec(sys, inter) = [inter.species_map[sys.atoms_data[i].element] for i in eachindex(sys.coords)]

function AtomsCalculators.potential_energy(sys::System, inter::Molly.AllegroPotential; kwargs...)
    coords_A = _coords_to_angstrom(sys.coords)
    species = _species_vec(sys, inter)
    bdy = _boundary_to_angstrom(sys.boundary)
    E = Molly.allegro_total_energy(inter.model, coords_A, species, bdy, inter.model.r_c)
    return _allegro_energy_to_units(E, sys.energy_units)
end

function AtomsCalculators.forces!(fs, sys::System, inter::Molly.AllegroPotential; kwargs...)
    coords_A = _coords_to_angstrom(sys.coords)
    species = _species_vec(sys, inter)
    bdy = _boundary_to_angstrom(sys.boundary)
    # forces from the model are in eV/Å (energy eV, length Å); Molly coords are nm, so the force
    # per nm-coordinate is 10× the per-Å force (dÅ/dnm = 10).
    F = Molly.allegro_forces(inter.model, coords_A, species, bdy, inter.model.r_c)
    for i in eachindex(fs)
        fs[i] += _allegro_force_to_units(F[i] .* 10, sys.force_units)
    end
    return fs
end

end # module
