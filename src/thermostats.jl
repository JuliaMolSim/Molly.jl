# Thermostats

export
    AndersenThermostat,
    apply_thermostat!,
    NoThermostat,
    velocity,
    maxwellboltzmann,
    temperature

"Rescale random velocities according to the Andersen thermostat."
struct AndersenThermostat <: Thermostat
    coupling_const::Float64
end

"Apply a thermostat to modify a simulation."
function apply_thermostat!(s::Simulation, thermostat::AndersenThermostat)
    dims = length(first(s.velocities))
    for i in 1:length(s.velocities)
        if rand() < s.timestep / thermostat.coupling_const
            mass = s.atoms[i].mass
            s.velocities[i] = velocity(mass, s.temperature; dims=dims)
        end
    end
    return s
end

"Placeholder thermostat that does nothing."
struct NoThermostat <: Thermostat end

function apply_thermostat!(s::Simulation, ::NoThermostat)
    return s
end

"Generate a random velocity from the Maxwell-Boltzmann distribution."
function velocity(mass::Real, T::Real; dims::Integer=3)
    return SVector([maxwellboltzmann(mass, T) for i in 1:dims]...)
end

"Draw from the Maxwell-Boltzmann distribution."
function maxwellboltzmann(mass::Real, T::Real)
    return rand(Normal(0.0, sqrt(T / mass)))
end

"Calculate the temperature of a system from the kinetic energy of the atoms."
function temperature(s::Simulation)
    ke = sum([a.mass * dot(s.velocities[i], s.velocities[i]) for (i, a) in enumerate(s.atoms)]) / 2
    df = 3 * length(s.coords) - 3
    return 2 * ke / df
end
