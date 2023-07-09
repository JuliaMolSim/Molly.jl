@derived_dimension MolarMass Unitful.𝐌/Unitful.𝐍 true

function check_system_units(masses, coords, velocities, energy_units, force_units)
    
    length_dim = validate_coords(coords)
    vel_dim = validate_velocities(velocities)
    force_dim = dimension(force_units)
    energy_dim = dimension(energy_units)
    mass_dim = validate_masses(masses)
    validate_energy_units(energy_units)

    forceIsMolar = #TODO Remove force_untis?
    energyIsMolar = (energy_dim == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2")
    massIsMolar = (mass_dim == u"𝐌* 𝐍^-1")
    

    if energyIsMolar == massIsMolar
        throw(ArgumentError("System was constructed with inconsistent energy & mass units. Both must be molar, non-molar or unitless.
            For example, kcal & kg are allowed but kcal/mol and kg is not allowed."))
    end

    allNoDims = all([length_dim, vel_dim, energy_dim, mass_dim] .== NoDims)
    anyNoDims = any([length_dim, vel_dim, energy_dim, mass_dim] .== NoDims)

    # If something has NoDims, all other data must have NoDims
    if anyNoDims && !allNoDims
        throw(ArgumentError("Either coords, velocities, masses or energy_units has NoDims/NoUnits but
            the others do have units. Molly does not permit mixing dimensionless and dimensioned data."))
    end

        #TODO: Choose correct version of Boltzmann Constnat

    #TODO: CHeck units passed to interactions & system
        #TODO: Can i remove the need to pass this to both
        #TODO: Change passed units to energy, length, mass? (infer in this func?)
end


function validate_energy_units(energy_units)
    valid_energy_dimensions = [u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2", u"𝐋^2 * 𝐌 * 𝐓^-2", NoDims]
    if dimension(energy_units) ∉ valid_energy_dimensions
        throw(ArgumentError("$(energy_units) are not energy units. Energy units must be energy,
            energy/amount, or NoUnits. For example, kcal & kcal/mol"))
    end
end

function validate_masses(masses)
    mass_units = unit.(masses)

    if !allequal(mass_units)
        throw(ArgumentError("Atoms array constructed with mixed mass units"))
    end

    valid_mass_dimensions = [u"𝐌", u"𝐌* 𝐍^-1", NoDims]
    mass_dimension = dimension(masses[1])

    if mass_dimension ∉ valid_mass_dimensions
        throw(ArgumentError("$(mass_dimension) are not mass units. Mass units must be mass or 
            mass/amount or NoUnits. For example, 1.0u\"kg\", 1.0u\"kg/mol\", & 1.0 are valid masses."))
    end

    return mass_dimension
end

function validate_coords(coords)
    coord_units = unit.(coords)

    if !allequal(coord_units)
        throw(ArgumentError("Atoms array constructed with mixed length units"))
    end

    valid_length_dimensions = [u"𝐋", NoDims]
    coord_dimension = dimension(coords[1][1])

    if coord_dimension ∉ valid_length_dimensions
        throw(ArgumentError("$(coord_dimension) are not length units. Length units must be length or 
            or NoUnits. For example, 1.0u\"m\" & 1.0 are valid positions."))
    end

    return coord_dimension
end

function validate_velocities(velocities)
    velocity_units = unit.(velocities)

    if !allequal(velocity_units)
        throw(ArgumentError("Velocities have mixed units"))
    end

    valid_velocity_dimensions = [u"𝐋 * 𝐓^-1", NoDims]
    velocity_dimension = dimension(velocities[1][1])

    if velocity_dimension ∉ valid_velocity_dimensions
        throw(ArgumentError("$(velocity_dimension) are not velocity units. Velocity units must be velocity or 
            or NoUnits. For example, 1.0u\"m/s\" & 1.0 are valid velocities."))
    end

    return velocity_dimension
end

# Convert the Boltzmann constant k to suitable units and float type
# Assumes temperature untis are Kelvin
function convert_k_units(T, k, energy_units)
    if energy_units == NoUnits
        if unit(k) == NoUnits
            k_converted = T(k)
        else
            throw(ArgumentError("energy_units was passed as NoUnits but units were provided on k: $(unit(k))"))
        end
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2" # Energy / Amount
        k_converted = T(uconvert(energy_units * u"K^-1", k * Unitful.Na))
    else
        k_converted = T(uconvert(energy_units * u"K^-1", k))
    end
    return k_converted
end


function check_energy_units(E, energy_units)
    if unit(E) != energy_units
        error("system energy units are ", energy_units, " but encountered energy units ",
                unit(E))
    end
end

#TODO THESE SHOULD NOT BE NECESSARY ANYMORE
function energy_remove_mol(x)
    if dimension(x) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(x))
        return x / T(Unitful.Na)
    else
        return x
    end
end

function energy_add_mol(x, energy_units)
    if dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(x))
        return x * T(Unitful.Na)
    else
        return x
    end
end