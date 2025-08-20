module MollyLAMMPSExt

using LAMMPS
using Molly
using Unitful
import AtomsCalculators
import AtomsBase

const lammps_mass_units_map = Dict("metal" => u"g/mol", "real" => u"g/mol", "si" => u"kg")

function convert_mass(m, lammps_units)
    sys_mass_unit = unit(m)
    lammps_mass_units = get(lammps_mass_units_map, lammps_units, sys_mass_unit)
    lammps_mass_is_molar = (dimension(lammps_mass_units) == u"𝐌* 𝐍^-1")
    sys_mass_is_molar = (dimension(sys_mass_unit) == u"𝐌* 𝐍^-1")

    lammps_molar_sys_not = (lammps_mass_is_molar && (!sys_mass_is_molar))
    sys_molar_lammps_not = (sys_mass_is_molar && (!lammps_mass_is_molar))

    if lammps_molar_sys_not
        return ustrip.(lammps_mass_units, Unitful.Na * m)
    elseif sys_molar_lammps_not
        return ustrip.(lammps_mass_units, m / Unitful.Na) 
    else # both molar or both non-molar
        return ustrip.(lammps_mass_units, m)
    end
end

function check_lammps_units(energy_unit, force_unit, lammps_units)

    length_unit = inv(force_unit / energy_unit)

    err = (U, LE, LA, EE, EA) -> ArgumentError("You picked $(U) units. Expected length units $(LE) and got $(LA). Expected energy units $(EE) and got $(EA)")

    if lammps_units == "metal"
        if length_unit != u"angstrom" && energy_unit != u"eV"
            error(err("metal", u"angstrom", length_unit, u"eV", energy_unit))
        end
    elseif lammps_units == "lj"
        if length_unit != NoUnits && energy_unit != NoUnits
            error(err("lj", "NoUnits", length_unit, "NoUnits", energy_unit))
        end
    elseif lammps_units == "real"
        if length_unit != u"angstrom" && energy_unit != u"kcal/mol"
            error(err("real", u"angstrom", length_unit, u"kcal/mol", energy_unit))
        end
    elseif lammps_units == "si"
        if length_unit != u"m" && energy_unit != u"J"
            error(err("si", u"m", length_unit, u"J", energy_unit))
        end
    else
        error(ArgumentError("Unsupported LAMMPS unit system, $(lammps_units). Expected one of (metal, real, lj)."))
    end

end

#! TODO: POTENTIAL DEFINITIONS TYPICALLY USE TYPES
#! BUT I AM ASSIGNING THOSE AUTOMAGICALLY
function Molly.LAMMPSCalculator(
        sys::System{3, AT, T},
        lammps_unit_system::String,
        potential_definition::Union{String, Array{String}};
        n_threads::Integer=Threads.nthreads(),
        extra_lammps_commands::Union{String, Array{String}} = "",
        logfile_path::String = "none",
        calculate_potential::Bool = false
    ) where {AT, T}

    # check that we're on CPU
    if AT != Array
        error(ArgumentError("LAMMPSCalculator only supports CPU execution."))
    end

    if sys.boundary == TriclinicBoundary
        error(ArgumentError("LAMMPSCalculator does not support triclinic systems yet. PRs welcome :)"))
    end

    if Molly.has_infinite_boundary(sys.boundary)
        error(ArgumentError("LAMMPSCalculator does not support systems with infinite boundaries. Must be fully periodic. PRs welcome :)"))
    end

    if length(potential_definition) == 0
        error(ArgumentError("Cannot pass emptry string as potential definition for LAMMPSCalculator"))
    end

    if T != Float64
        @warn "LAMMPS uses Float64, you are using $T. You might incur a penalty from type promotion."
    end

    check_lammps_units(sys.energy_units, sys.force_units, lammps_unit_system)

    # If Molly ever uses MPI this can take the MPI.Comm handle
    # For now just use OpenMP
    lmp = LMP(["-screen","none", "-sf", "omp", "-pk", "omp", "$(n_threads)"])

    all_syms = Molly.atomic_symbol(sys)
    unique_syms = unique(all_syms)

    if any(unique_syms .== :unknown)
        error(ArgumentError("All atoms must have atomic symbols to use LAMMPSCalculators"))
    end

    unique_sym_idxs = Dict(sym => findfirst(x -> x == sym, all_syms) for sym in unique_syms)
    type_map = Dict(sym => Int32(i) for (i, sym) in enumerate(unique_syms))
    types = [type_map[sym] for sym in all_syms]
    ids = collect(Int32, 1:length(sys))

    command(lmp, """
            log $(logfile_path)
            units $(lammps_unit_system)
            atom_style atomic
            atom_modify map array sort 0 0
        """)

    xhi, yhi, zhi = ustrip.(sys.boundary)
    command(lmp, """
            boundary p p p
            region cell block 0 $(xhi) 0 $(yhi) 0 $(zhi) units box
            create_box $(length(unique_syms)) cell
        """)

    LAMMPS.create_atoms(
        lmp,
        reinterpret(reshape, Float64, sys.coords),
        ids,
        types
    )   

    m_lmp = Dict(type_map[sym] => convert_mass(sys.masses[i], lammps_unit_system) for (sym, i) in unique_sym_idxs)
    command(lmp, ["mass $(type) $(m)" for (type,m) in  m_lmp])

    command(lmp, potential_definition)

    if length(extra_lammps_commands) > 0
        command(lmp, extra_lammps_commands)
    end

    #! NONE OF THESE ARE ACTUALLY TYPES DOES NOT WORK....
    # calculate_potential |=  any(x -> isa(x, PotentialEnergyLogger) || isa(x, TotalEnergyLogger), sys.loggers)
    calculate_potential && command(lmp, "compute pot_e all pe")

    return LAMMPSCalculator{typeof(lmp)}(lmp, -1)
end

function maybe_run_lammps_calc!(lammps_calc, r::AbstractVector{T}, step_n) where T
    if lammps_calc.last_updated != step_n
        # Send current coordinates to LAMMPS
        scatter!(lammps_calc.lmp, "x", reinterpret(reshape, Float64, r))
        # Run a step, will execute the registered computes/fixes
        command(lammps_calc.lmp, "run 0")
        lammps_calc.last_updated = step_n
    end
end

function AtomsCalculators.forces!(fs::AbstractVector{T}, sys, inter::LAMMPSCalculator; step_n=0, kwargs...) where T
    maybe_run_lammps_calc!(inter, sys.coords, step_n)
    gather!(inter.lmp, "f", reinterpret(reshape, Float64, fs)) 
    return fs
end

function AtomsCalculators.potential_energy(sys, inter::LAMMPSCalculator; step_n=0, kwargs...)
    maybe_run_lammps_calc!(inter, sys.coords, step_n)
    return extract_compute(inter.lmp, "pot_e", STYLE_GLOBAL, TYPE_SCALAR)[1] * sys.energy_units
end


end # module MollyLAMMPSExt


