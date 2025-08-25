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
        if length_unit != u"angstrom" || energy_unit != u"eV"
            error(err("metal", u"angstrom", length_unit, u"eV", energy_unit))
        end
    elseif lammps_units == "lj"
        if length_unit != NoUnits || energy_unit != NoUnits
            error(err("lj", "NoUnits", length_unit, "NoUnits", energy_unit))
        end
    elseif lammps_units == "real"
        if length_unit != u"angstrom" || energy_unit != u"kcal/mol"
            error(err("real", u"angstrom", length_unit, u"kcal/mol", energy_unit))
        end
    elseif lammps_units == "si"
        if length_unit != u"m" || energy_unit != u"J"
            error(err("si", u"m", length_unit, u"J", energy_unit))
        end
    else
        error(ArgumentError("Unsupported LAMMPS unit system, $(lammps_units). Expected one of (metal, real, lj, si)."))
    end

end

function has_logger_with_pe(sys)
    for logger in sys.loggers
        if logger.observable == Molly.potential_energy_wrapper || logger.observable == Molly.total_energy_wrapper
            return true
        end
    end
    return false
end

function Molly.LAMMPSCalculator(
        sys::System{3, AT, T},
        lammps_unit_system::String,
        potential_definition::Union{String, Array{String}};
        label_type_map::Dict{Symbol, Int} = Dict{Symbol, Int}(),
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

    # OpenMP doesnt seem to make a difference...
    # lmp = LMP(["-screen","none", "-sf", "omp", "-pk", "omp", "$(n_threads)"], LAMMPS.MPI.COMM_WORLD)
    lmp = LMP(["-screen","none"], LAMMPS.MPI.COMM_WORLD)

    all_syms = Molly.atomic_symbol(sys)
    unique_syms = unique(all_syms)
    unique_sym_idxs = Dict(sym => findfirst(x -> x == sym, all_syms) for sym in unique_syms)

    if any(unique_syms .== :unknown)
        error(ArgumentError("All atoms must have atomic symbols to use LAMMPSCalculator"))
    end
    
    ids = collect(Int32, 1:length(sys))
    xhi, yhi, zhi = ustrip.(sys.boundary)

    if length(label_type_map) == 0
        label_type_map = Dict(sym => Int32(i) for (i, sym) in enumerate(unique_syms))
        types = [label_type_map[sym] for sym in all_syms]
    else 
        unique_sym_user = keys(label_type_map)
        if Set(unique_sym_user) != Set(unique_syms)
            error(ArgumentError("You provided a label_type_map with $(unique_sym_user) symbols, but" *
                " the system has $(unique_syms). They must match exactly if you pass label_type_map."))
        end
        types = [Int32(label_type_map[sym]) for sym in all_syms]
    end

    m_lmp = Dict(label_type_map[sym] => convert_mass(sys.masses[i], lammps_unit_system) for (sym, i) in unique_sym_idxs)

    label_map_cmd = "labelmap atom " * join(["$(i) $(sym)" for (sym,i) in label_type_map], " ") 

    setup_cmd = """
            log $(logfile_path)
            units $(lammps_unit_system)
            atom_style atomic
            atom_modify map array sort 0 0
        """
    
    cell_cmd = """
            boundary p p p
            region cell block 0 $(xhi) 0 $(yhi) 0 $(zhi) units box
            create_box $(length(unique_syms)) cell
            $(label_map_cmd)
        """
    
    mass_cmd = join(["mass $(type) $(m)" for (type,m) in  m_lmp], "\n")

    if length(extra_lammps_commands) > 0
        command(lmp, extra_lammps_commands)
    end

    #! NONE OF THESE ARE ACTUALLY TYPES DOES NOT WORK....
    calculate_potential |= has_logger_with_pe(sys)
    if calculate_potential
        command(lmp, "compute pot_e all pe")
    end

    command(lmp, setup_cmd)
    command(lmp, cell_cmd)
    command(lmp, mass_cmd)

    LAMMPS.create_atoms(
        lmp,
        reinterpret(reshape, Float64, sys.coords),
        ids,
        types
    )   

    try
        command(lmp, potential_definition)
    catch e
        if startswith(e.msg, "Number of element to type mappings does")
            @info "Ensure path to potential definition is wrapped in quotes if there are spaces in path."
        end
        rethrow(e)
    end

    # This allows LAMMPS to register the computes/fixes
    # and build the neighbor list. 
    command(lmp, "run 0 post no")

    return LAMMPSCalculator{typeof(lmp)}(lmp, -1)
end

function maybe_run_lammps_calc!(lammps_calc, r::AbstractVector{T}, step_n) where T
    if lammps_calc.last_updated != step_n
        # Send current coordinates to LAMMPS
        scatter!(lammps_calc.lmp, "x", reinterpret(reshape, Float64, r))
        # Run a step, will execute the registered computes/fixes
        command(lammps_calc.lmp, "run 0 pre no post no")
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


