# Python ASE interface
# This file is only loaded when PythonCall is imported

module MollyPythonCallExt

using Molly
using PythonCall
import AtomsCalculators
using GPUArrays
using StaticArrays
using Unitful

# See PythonCall precompilation documentation
const ase = Ref{Py}()

function __init__()
    try
        ase[] = pyimport("ase")
    catch
        @warn "MollyPythonCallExt is loaded but the ase Python package is not available, " *
              "ASECalculator will not work"
    end
end

inf_to_one(x) = isinf(x) ? one(x) : x
inf_to_flag(x) = isinf(x) ? 0 : 1

function Molly.ASECalculator(;
                             ase_calc,
                             atoms,
                             coords,
                             boundary,
                             elements=nothing,
                             atoms_data=nothing,
                             velocities=nothing)
    if !isnothing(elements)
        element_string = (elements isa AbstractString ? elements : join(elements))
    elseif !isnothing(atoms_data)
        element_string = join(ad.element for ad in atoms_data)
    else
        throw(ArgumentError("either elements or atoms_data must be provided to ASECalculator"))
    end
    atoms_cpu = Array(atoms)
    if unit(boundary.side_lengths[1]) == NoUnits
        # Assume units are ASE units
        coords_strip = pylist(Array(coords))
        box = boundary.side_lengths
        masses_strip = pylist(mass.(atoms_cpu))
        velocities_strip = (isnothing(velocities) ? pybuiltins.None : pylist(Array(velocities)))
    else
        coords_strip = pylist(ustrip_vec.(u"Å", Array(coords)))
        box = ustrip.(u"Å", boundary.side_lengths)
        if dimension(mass(first(atoms_cpu))) == u"𝐌 * 𝐍^-1"
            masses_nomol = mass.(atoms_cpu) / Unitful.Na
        else
            masses_nomol = mass.(atoms_cpu)
        end
        masses_strip = pylist(ustrip.(u"u", masses_nomol))
        velocities_strip = (isnothing(velocities) ? pybuiltins.None :
                                    pylist(ustrip_vec.(u"u^(-1/2) * eV^(1/2)", Array(velocities))))
    end
    ase_atoms = ase[].Atoms(
        element_string,
        positions=coords_strip,
        cell=pylist(inf_to_one.(box)),
        pbc=pylist(inf_to_flag.(box)),
        masses=masses_strip,
        charges=pylist(charge.(atoms_cpu)),
        velocities=velocities_strip,
    )
    ase_atoms.calc = ase_calc
    return ASECalculator(ase_atoms, ase_calc)
end

function Molly.update_ase_calc!(ase_calc, sys)
    if unit(sys.boundary.side_lengths[1]) == NoUnits
        # Assume units are ASE units
        coords_current = pylist(Array(sys.coords))
        velocities_current = pylist(Array(sys.velocities))
        box = sys.boundary.side_lengths
    else
        coords_current = pylist(ustrip_vec.(u"Å", Array(sys.coords)))
        velocities_current = pylist(ustrip_vec.(u"u^(-1/2) * eV^(1/2)", Array(sys.velocities)))
        box = ustrip.(u"Å", sys.boundary.side_lengths)
    end
    ase_calc.ase_atoms.set_positions(coords_current)
    ase_calc.ase_atoms.set_velocities(velocities_current)
    ase_calc.ase_atoms.set_cell(pylist(inf_to_one.(box)))
    return ase_calc
end

uconvert_vec(x...) = uconvert.(x...)

function AtomsCalculators.forces(sys::System{D, AT, T},
                                 ase_calc::ASECalculator;
                                 kwargs...) where {D, AT, T}
    Molly.update_ase_calc!(ase_calc, sys)
    forces_py = ase_calc.ase_atoms.get_forces()
    forces_flat = reshape(transpose(pyconvert(Matrix{T}, forces_py)), length(sys) * D)
    fs = reinterpret(SVector{D, T}, forces_flat)
    if sys.force_units == NoUnits
        fs_unit = fs # Assume units are eV/Å
    elseif dimension(sys.force_units) == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        fs_unit = uconvert_vec.(sys.force_units, fs * Unitful.Na * u"eV/Å")
    else
        fs_unit = uconvert_vec.(sys.force_units, fs * u"eV/Å")
    end
    return AT(fs_unit)
end

function AtomsCalculators.potential_energy(sys::System{D, AT, T},
                                           ase_calc::ASECalculator;
                                           kwargs...) where {D, AT, T}
    Molly.update_ase_calc!(ase_calc, sys)
    pe_py = ase_calc.ase_atoms.get_potential_energy()
    pe = pyconvert(T, pe_py)
    if sys.energy_units == NoUnits
        return pe # Assume units are eV
    elseif dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert(sys.energy_units, pe * Unitful.Na * u"eV")
    else
        return uconvert(sys.energy_units, pe * u"eV")
    end
end

end
