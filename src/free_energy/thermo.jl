export 
    ThermoState

"""
    ThermoState(system::System, integrator; name=nothing)

Thermodynamic state wrapper carrying the system, integrator, and derived thermodynamic properties 
(inverse temperature `β` and pressure `p`).

# Arguments
- `system::System`: The simulation system used to evaluate energies.
- `integrator`: The integrator used to simulate the system. Must define temperature and/or pressure couplings 
    (e.g., [`Langevin`](@ref), [`VelocityVerlet`](@ref) with thermostat/barostat).
- `name::AbstractString=nothing`: A label for the state. If not provided, a default name based on 
    temperature and pressure is generated.
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

function logsumexp(x::AbstractVector{T}) where T
    isempty(x) && return -T(Inf)
    x_max = maximum(x)
    # If all weights are -Inf (e.g. huge energy overlap issues), return -Inf
    !isfinite(x_max) && return x_max 
    
    s = zero(T)
    for val in x
        s += exp(val - x_max)
    end
    return x_max + log(s)
end