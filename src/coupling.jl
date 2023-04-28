# Temperature and pressure coupling methods

export
    apply_coupling!,
    NoCoupling,
    AndersenThermostat,
    RescaleThermostat,
    BerendsenThermostat

"""
    apply_coupling!(system, coupling, simulator, neighbors=nothing,
                    step_n=0; n_threads=Threads.nthreads())

Apply a coupler to modify a simulation.
Returns whether the coupling has invalidated the currently stored forces, for
example by changing the coordinates.
If `coupling` is a tuple or named tuple then each coupler will be applied in turn.
Custom couplers should implement this function.
"""
function apply_coupling!(sys, couplers::Union{Tuple, NamedTuple}, sim, neighbors,
                         step_n; kwargs...)
    recompute_forces = false
    for coupler in couplers
        rf = apply_coupling!(sys, coupler, sim, neighbors, step_n; kwargs...)
        if rf
            recompute_forces = true
        end
    end
    return recompute_forces
end

"""
    NoCoupling()

Placeholder coupler that does nothing.
"""
struct NoCoupling end

apply_coupling!(sys, ::NoCoupling, sim, neighbors, step_n; kwargs...) = false

"""
    AndersenThermostat(temperature, coupling_const)

Rescale random velocities according to the Andersen thermostat.
"""
struct AndersenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys::System{D, false}, thermostat::AndersenThermostat, sim,
                         neighbors=nothing, step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads()) where D
    for i in eachindex(sys)
        if rand() < (sim.dt / thermostat.coupling_const)
            sys.velocities[i] = random_velocity(mass(sys.atoms[i]), thermostat.temperature, sys.k;
                                                dims=n_dimensions(sys))
        end
    end
    return false
end

function apply_coupling!(sys::System{D, true, T}, thermostat::AndersenThermostat, sim,
                         neighbors=nothing, step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads()) where {D, T}
    atoms_to_bump = T.(rand(length(sys)) .< (sim.dt / thermostat.coupling_const))
    atoms_to_leave = one(T) .- atoms_to_bump
    atoms_to_bump_dev = move_array(atoms_to_bump, sys)
    atoms_to_leave_dev = move_array(atoms_to_leave, sys)
    vs = random_velocities(sys, thermostat.temperature)
    sys.velocities = sys.velocities .* atoms_to_leave_dev .+ vs .* atoms_to_bump_dev
    return false
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

function apply_coupling!(sys, thermostat::RescaleThermostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    sys.velocities *= sqrt(thermostat.temperature / temperature(sys))
    return false
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

function apply_coupling!(sys, thermostat::BerendsenThermostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    λ2 = 1 + (sim.dt / thermostat.coupling_const) * ((thermostat.temperature / temperature(sys)) - 1)
    sys.velocities *= sqrt(λ2)
    return false
end
