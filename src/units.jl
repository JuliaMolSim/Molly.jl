export ustrip_vec

# Unit types to dispatch on
@derived_dimension MolarMass Unitful.𝐌/Unitful.𝐍 true
@derived_dimension BoltzmannConstUnits Unitful.𝐌*Unitful.𝐋^2*Unitful.𝐓^-2*Unitful.𝚯^-1 true
@derived_dimension MolarBoltzmannConstUnits Unitful.𝐌*Unitful.𝐋^2*Unitful.𝐓^-2*Unitful.𝚯^-1*Unitful.𝐍^-1 true

"""
    ustrip_vec(x)
    ustrip_vec(u, x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustrip_vec.(coords)`.
"""
ustrip_vec(x...) = ustrip.(x...)

# Parses the length, mass, velocity, energy and force units and verifies they are
#   correct and consistent with other parameters passed to the system.
function check_units(atoms, coords, velocities, energy_units, force_units,
                     p_inters, s_inters, g_inters, boundary)
    masses = mass.(atoms)
    sys_units = check_system_units(masses, coords, velocities, energy_units, force_units)
    check_other_units(atoms, boundary, sys_units)
    return sys_units
end

function check_system_units(masses, coords, velocities, energy_units, force_units)
    length_dim, length_units = validate_coords(coords)
    vel_dim, vel_units = validate_velocities(velocities)
    force_dim = dimension(force_units)
    energy_dim = dimension(energy_units)
    mass_dim, mass_units = validate_masses(masses)
    validate_energy_units(energy_units)

    force_is_molar = (force_dim == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2")
    energy_is_molar = (energy_dim == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2")
    mass_is_molar = (mass_dim == u"𝐌* 𝐍^-1")

    if !(energy_is_molar == mass_is_molar && energy_is_molar == force_is_molar)
        throw(ArgumentError("System was constructed with inconsistent energy, force and mass " *
            "units. All must be molar, non-molar or unitless. For example, kcal and kg is " *
            "allowed but kcal/mol and kg is not. Units were $([energy_units, mass_units, force_units])"))
    end

    no_dim_arr = [dim == NoDims for dim in [length_dim, vel_dim, energy_dim, force_dim, mass_dim]]

    # If something has NoDims, all other data must have NoDims
    if any(no_dim_arr) && !all(no_dim_arr)
        throw(ArgumentError("either coords, velocities, masses or energy_units has " *
            "NoDims/NoUnits but the others do have units. Molly does not permit mixing " *
            "data with and without units."))
    end

    # Check derived units
    if force_units != (energy_units / length_units)
        throw(ArgumentError("force_units was specified as $force_units, but that is " *
            "different from energy_units divided by the coordinate length units"))
    end

    return NamedTuple{(:length, :velocity, :mass, :energy, :force)}((length_units,
        vel_units, mass_units, energy_units, force_units))
end

function check_other_units(atoms_dev, boundary, sys_units::NamedTuple)
    atoms = Array(atoms_dev)
    box_units = unit(length_type(boundary))

    if !all(sys_units[:length] .== box_units)
        throw(ArgumentError("simulation box constructed with $box_units but length unit on coords was $(sys_units[:length])"))
    end

    sigmas   = getproperty.(atoms[hasproperty.(atoms, :σ)], :σ)
    epsilons = getproperty.(atoms[hasproperty.(atoms, :ϵ)], :ϵ)

    if !all(sigmas .== 0.0u"nm")
        σ_units = unit.(sigmas)
        if !all(sys_units[:length] .== σ_units)
            throw(ArgumentError("Atom σ has $(σ_units[1]) units but length unit on coords was $(sys_units[:length])"))
        end
    end

    if !all(epsilons .== 0.0u"kJ * mol^-1")
        ϵ_units = unit.(epsilons)
        if !all(sys_units[:energy] .== ϵ_units)
            throw(ArgumentError("Atom ϵ has $(ϵ_units[1]) units but system energy unit was $(sys_units[:energy])"))
        end
    end
end

function validate_energy_units(energy_units)
    valid_energy_dimensions = [u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2", u"𝐋^2 * 𝐌 * 𝐓^-2", NoDims]
    if dimension(energy_units) ∉ valid_energy_dimensions
        throw(ArgumentError("$energy_units do not have dimensions of energy. Energy units must " *
            "be energy, energy/number, or NoUnits, e.g. kcal or kcal/mol."))
    end
end

function validate_masses(masses)
    mass_units = unit.(Array(masses))
    if !allequal(mass_units)
        throw(ArgumentError("atom array constructed with mixed mass units"))
    end

    valid_mass_dimensions = [u"𝐌", u"𝐌* 𝐍^-1", NoDims]
    mass_dimension = dimension(eltype(masses))

    if mass_dimension ∉ valid_mass_dimensions
        throw(ArgumentError("mass units have dimension $mass_dimension. Mass units must be " *
            "mass, mass/number or NoUnits, e.g. 1.0u\"kg\", 1.0u\"kg/mol\" or 1.0."))
    end

    return mass_dimension, mass_units[1]
end

function validate_coords(coords)
    coord_units = map(Array(coords)) do coord
        [unit(c) for c in coord]
    end

    if !allequal(coord_units) || !allequal(coord_units[1])
        throw(ArgumentError("coordinates have mixed units"))
    end

    valid_length_dimensions = [u"𝐋", NoDims]
    coord_dimension = (dimension ∘ eltype ∘ eltype)(coords)

    if coord_dimension ∉ valid_length_dimensions
        throw(ArgumentError("coordinate units have dimension $coord_dimension. Length units " *
            "must be length or NoUnits, e.g. 1.0u\"m\" or 1.0."))
    end

    return coord_dimension, coord_units[1][1]
end

function validate_velocities(velocities)
    velocity_units = map(Array(velocities)) do vel
        [unit(v) for v in vel]
    end

    if !allequal(velocity_units) || !allequal(velocity_units[1])
        throw(ArgumentError("velocities have mixed units"))
    end

    valid_velocity_dimensions = [u"𝐋 * 𝐓^-1", NoDims]
    velocity_dimension = (dimension ∘ eltype ∘ eltype)(velocities)

    if velocity_dimension ∉ valid_velocity_dimensions
        throw(ArgumentError("velocity units have dimension $velocity_dimension. Velocity units " *
            "must be velocity or NoUnits, e.g. 1.0u\"m/s\" or 1.0."))
    end

    return velocity_dimension, velocity_units[1][1]
end

function default_k(energy_units)
    if dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k = Unitful.k * Unitful.Na
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐓^-2"
        k = Unitful.k
    elseif energy_units == NoUnits
        k = ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)
    else
        throw(ArgumentError("energy units do not have dimensions of energy: $energy_units"))
    end

    return k
end

# Convert the Boltzmann constant k to suitable units and float type
# Assumes temperature units are Kelvin
function convert_k_units(T, k, energy_units)
    if energy_units == NoUnits
        if unit(k) == NoUnits
            # Use user-supplied unitless Boltzmann constant
            k_converted = T(k)
        else
            @warn "Units will be stripped from Boltzmann constant: energy_units was passed as NoUnits and units were provided on k: $(unit(k))"
            k_converted = T(ustrip(k))
        end
    elseif dimension(energy_units) in (u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2", u"𝐋^2 * 𝐌 * 𝐓^-2")
        if dimension(energy_units * u"K^-1") != dimension(k)
            throw(ArgumentError("energy_units ($energy_units) in System and Boltzmann constant units ($(unit(k))) are incompatible"))
        end
        k_converted = T(uconvert(energy_units * u"K^-1", k))
    else
        throw(ArgumentError("energy units do not have dimensions of energy: $energy_units"))
    end
    return k_converted
end

function check_energy_units(E, energy_units)
    if unit(E) != energy_units
        error("system energy units are ", energy_units, " but encountered energy units ", unit(E))
    end
end

function check_force_units(F, force_units)
    if unit(F) != force_units
        error("system force units are ", force_units, " but encountered force units ", unit(F))
    end
end

check_force_units(F::SVector, force_units) = @inbounds check_force_units(F[1], force_units)

function energy_remove_mol(x)
    if dimension(x) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(x))
        return x / T(Unitful.Na)
    else
        return x
    end
end
