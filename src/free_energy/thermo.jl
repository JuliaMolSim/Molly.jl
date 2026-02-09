export 
    ThermoState

"""
    ThermoState(name::AbstractString, β, p, system)
    ThermoState(system::System, β, p; name::Union{Nothing, AbstractString}=nothing)

Thermodynamic state wrapper carrying inverse temperature `β = 1/kBT`, pressure `p`,
and the [`System`](@ref) used to evaluate energies.

Fields:
- `name::String` - label for the state.
- `t_couple` - Temperature coupling scheme. If NoCoupling is used, make sure to use an
integrator that implicitly accounts for temperature, e.g., Langevin.
- `p_couple` - Pressure coupling scheme. Can be NoCoupling.
- `β` - inverse temperature with units compatible with `1/system.energy_units`.
- `system::System` - simulation system used to compute potential energy.

The second constructor checks unit consistency for `β` and `p` and sets a default
`name` when not provided.
"""
mutable struct ThermoState{I, B, P, S}
    name::String
    integrator::I # Integrator scheme
    beta::B      # Inverse temperature
    p::P         # Pressure
    system::S    # How to evaluate U_i on given coords and boundary
end

function ThermoState(sys::System{D, AT, FT}, integrator; 
    name::Union{Nothing, AbstractString}=nothing) where {D, AT, FT}

    temp_source = nothing
    press_source = nothing

    # We treat NoCoupling generally, skipping the loop if empty/irrelevant
    if !(integrator.coupling isa NoCoupling)
        for coupler in integrator.coupling
            if coupler isa AbstractThermostat
                temp_source = coupler.temperature
            elseif coupler isa AbstractBarostat
                press_source = coupler.pressure
            end
        end
    end

    # If no thermostat was found in couplings, check if the integrator itself controls T
    if isnothing(temp_source) && (integrator isa Langevin || integrator isa LangevinSplitting || integrator isa NoseHoover)
        temp_source = integrator.temperature
    end

    if isnothing(temp_source)
        throw(ArgumentError("No way was provided to maintain a constant temperature. " * "You must choose either an explicit thermostat or an " * "integrator with an implicit temperature coupling scheme."))
    end

    # Calculate Beta
    beta = try
        # Convert to Energy Units (1/kT)
        FT(1 / uconvert(sys.energy_units, Unitful.R * temp_source))
    catch
        throw(ArgumentError("Temperature provided is not compatible with system energy units."))
    end

    # Calculate Pressure
    press = isnothing(press_source) ? nothing : FT(1/3 * tr(press_source))

    # Fixed the 'pressure' vs 'press' bug here
    final_name = isnothing(name) ? "system_$(beta)_$(press)" : name
    
    return ThermoState(final_name, integrator, beta, press, sys)
end