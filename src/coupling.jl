# Temperature and pressure coupling

export
    NoCoupling,
    apply_coupling!,
    AndersenThermostat,
    RescaleThermostat,
    FrictionThermostat

"""
    NoCoupling()

Placeholder coupler that does nothing.
"""
struct NoCoupling end

"""
    apply_coupling!(system, simulator, coupling)

Apply a coupler to modify a simulation.
Custom couplers should implement this function.
"""
apply_coupling!(sys::System, sim, ::NoCoupling) = sys

"""
    AndersenThermostat(temperature, coupling_const)

Rescale random velocities according to the Andersen thermostat.
"""
struct AndersenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys::System{D}, sim, thermostat::AndersenThermostat) where D
    for i in 1:length(sys)
        if rand() < (sim.dt / thermostat.coupling_const)
            mass = sys.atoms[i].mass
            sys.velocities[i] = velocity(mass, thermostat.temperature; dims=D)
        end
    end
    return sys
end

struct RescaleThermostat{T}
    temperature::T
end

function apply_coupling!(sys::System, sim, thermostat::RescaleThermostat)
    sys.velocities *= sqrt(thermostat.temperature / temperature(sys))
    return sys
end

struct FrictionThermostat{T}
    friction_const::T
end

function apply_coupling!(sys::System, sim, thermostat::FrictionThermostat)
    sys.velocities *= thermostat.friction_const
    return sys
end
