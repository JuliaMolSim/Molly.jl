# Temperature and pressure coupling methods

export
    NoCoupling,
    apply_coupling!,
    AndersenThermostat,
    RescaleThermostat,
    BerendsenThermostat

"""
    NoCoupling()

Placeholder coupler that does nothing.
"""
struct NoCoupling end

"""
    apply_coupling!(system, coupling, simulator)

Apply a coupler to modify a simulation.
Custom couplers should implement this function.
"""
apply_coupling!(sys, ::NoCoupling, sim) = sys

"""
    AndersenThermostat(temperature, coupling_const)

Rescale random velocities according to the Andersen thermostat.
"""
struct AndersenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys::System{D, false}, thermostat::AndersenThermostat, sim) where D
    for i in eachindex(sys)
        if rand() < (sim.dt / thermostat.coupling_const)
            sys.velocities[i] = random_velocity(mass(sys.atoms[i]), thermostat.temperature, sys.k;
                                                dims=n_dimensions(sys))
        end
    end
    return sys
end

function apply_coupling!(sys::System{D, true, T}, thermostat::AndersenThermostat, sim) where {D, T}
    atoms_to_bump = T.(rand(length(sys)) .< (sim.dt / thermostat.coupling_const))
    atoms_to_leave = one(T) .- atoms_to_bump
    atoms_to_bump_dev = move_array(atoms_to_bump, sys)
    atoms_to_leave_dev = move_array(atoms_to_leave, sys)
    vs = random_velocities(sys, thermostat.temperature)
    sys.velocities = sys.velocities .* atoms_to_leave_dev .+ vs .* atoms_to_bump_dev
    return sys
end

@doc raw"""
    RescaleThermostat(temperature)

The velocity rescaling thermostat that immediately rescales the velocities to
match a target temperature.
This thermostat should be used with caution as it can lead to simulation
artifacts.
The scaling factor for the velocities each step is
```math
\lambda = \sqrt{\frac{T_0}{T}}
```
"""
struct RescaleThermostat{T}
    temperature::T
end

function apply_coupling!(sys, thermostat::RescaleThermostat, sim)
    sys.velocities *= sqrt(thermostat.temperature / temperature(sys))
    return sys
end

@doc raw"""
    BerendsenThermostat(temperature, coupling_const)

The Berendsen thermostat.
This thermostat should be used with caution as it can lead to simulation
artifacts.
The scaling factor for the velocities each step is
```math
\lambda^2 = 1 + \frac{\delta t}{\tau} \left( \frac{T_0}{T} - 1 \right)
```
"""
struct BerendsenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys, thermostat::BerendsenThermostat, sim)
    λ2 = 1 + (sim.dt / thermostat.coupling_const) * ((thermostat.temperature / temperature(sys)) - 1)
    sys.velocities *= sqrt(λ2)
    return sys
end
