# Python ASE interface
# This file is only loaded when PythonCall is imported

module MollyPythonCallExt

using Molly
using Molly: from_device, to_device
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
strip_Å(x) = (unit(x) == NoUnits ? x : ustrip(u"Å", x)) # Assume no units means ASE units

# Cell and periodicity of a boundary as ASE expects them
ase_cell(b) = pylist(inf_to_one.(strip_Å.(b.side_lengths)))
ase_cell(b::TriclinicBoundary) = pylist([pylist(strip_Å.(bv)) for bv in b.basis_vectors])
ase_pbc(b) = pylist(inf_to_flag.(b.side_lengths))
ase_pbc(b::TriclinicBoundary) = pylist([1, 1, 1])

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
    atoms_cpu = from_device(atoms)
    if unit(Molly.length_type(boundary)) == NoUnits
        # Assume units are ASE units
        coords_strip = pylist(from_device(coords))
        masses_strip = pylist(mass.(atoms_cpu))
        velocities_strip = (isnothing(velocities) ? pybuiltins.None :
                            pylist(from_device(velocities)))
    else
        coords_strip = pylist(ustrip_vec.(u"Å", from_device(coords)))
        if dimension(mass(first(atoms_cpu))) == u"𝐌 * 𝐍^-1"
            masses_nomol = mass.(atoms_cpu) / Unitful.Na
        else
            masses_nomol = mass.(atoms_cpu)
        end
        masses_strip = pylist(ustrip.(u"u", masses_nomol))
        velocities_strip = (isnothing(velocities) ? pybuiltins.None :
                            pylist(ustrip_vec.(u"u^(-1/2) * eV^(1/2)", from_device(velocities))))
    end
    ase_atoms = ase[].Atoms(
        element_string,
        positions=coords_strip,
        cell=ase_cell(boundary),
        pbc=ase_pbc(boundary),
        masses=masses_strip,
        charges=pylist(charge.(atoms_cpu)),
        velocities=velocities_strip,
    )
    ase_atoms.calc = ase_calc
    return ASECalculator(ase_atoms, ase_calc)
end

function Molly.update_ase_calc!(ase_calc, sys::System{<:Any, <:Any, T}) where T
    if unit(Molly.length_type(sys.boundary)) == NoUnits
        # Assume units are ASE units
        coords_nounits, velocities_nounits = sys.coords, sys.velocities
    else
        coords_nounits = ustrip_vec.(u"Å", from_device(sys.coords))
        velocities_nounits = ustrip_vec.(u"u^(-1/2) * eV^(1/2)", from_device(sys.velocities))
    end
    coords_current = Py(Array(transpose(reshape(
                            reinterpret(T, coords_nounits), 3, length(sys))))).to_numpy()
    velocities_current = Py(Array(transpose(reshape(
                            reinterpret(T, velocities_nounits), 3, length(sys))))).to_numpy()
    ase_calc.ase_atoms.set_positions(coords_current)
    ase_calc.ase_atoms.set_velocities(velocities_current)
    ase_calc.ase_atoms.set_cell(ase_cell(sys.boundary))
    return ase_calc
end

uconvert_vec(x...) = uconvert.(x...)

function from_ase_energy(x, energy_units)
    if energy_units == NoUnits
        return x # Assume units are eV
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert.(energy_units, x * Unitful.Na * u"eV")
    else
        return uconvert.(energy_units, x * u"eV")
    end
end

function AtomsCalculators.forces!(fs,
                                  sys::System{D, AT, T},
                                  ase_calc::ASECalculator;
                                  buffers=nothing,
                                  needs_vir=false,
                                  strictness=Molly.default_strictness(),
                                  kwargs...) where {D, AT, T}
    Molly.update_ase_calc!(ase_calc, sys)
    forces_py = ase_calc.ase_atoms.get_forces()
    forces_flat = reshape(transpose(pyconvert(Matrix{T}, forces_py)), length(sys) * D)
    fs_svec = reinterpret(SVector{D, T}, forces_flat)
    if sys.force_units == NoUnits
        fs_unit = fs_svec # Assume units are eV/Å
    elseif dimension(sys.force_units) == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        fs_unit = uconvert_vec.(sys.force_units, fs_svec * Unitful.Na * u"eV/Å")
    else
        fs_unit = uconvert_vec.(sys.force_units, fs_svec * u"eV/Å")
    end
    fs .+= to_device(fs_unit, AT)
    if needs_vir
        # W = Σ r ⊗ f = -V σ, with the ASE stress σ in eV/Å^3
        stress = try
            pyconvert(Matrix{T}, ase_calc.ase_atoms.get_stress(voigt=false))
        catch e
            not_impl = pyimport("ase.calculators.calculator").PropertyNotImplementedError
            (e isa PyException && pyisinstance(e.v, not_impl)) || rethrow()
            Molly.report_issue("The ASE calculator does not provide the stress, so its " *
                               "virial contribution will be ignored", strictness; maxlog=1)
            zeros(T, D, D)
        end
        vir = -pyconvert(T, ase_calc.ase_atoms.get_volume()) .* stress
        buffers.virial .+= from_ase_energy(vir, sys.energy_units)
    end
    return fs
end

function AtomsCalculators.potential_energy(sys::System{<:Any, <:Any, T},
                                           ase_calc::ASECalculator;
                                           kwargs...) where T
    Molly.update_ase_calc!(ase_calc, sys)
    pe_py = ase_calc.ase_atoms.get_potential_energy()
    pe = pyconvert(T, pe_py)
    return from_ase_energy(pe, sys.energy_units)
end

end
