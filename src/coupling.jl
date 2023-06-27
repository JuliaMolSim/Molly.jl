# Temperature and pressure coupling methods

export
    apply_coupling!,
    NoCoupling,
    AndersenThermostat,
    RescaleThermostat,
    BerendsenThermostat,
    MonteCarloBarostat,
    MonteCarloAnisotropicBarostat

"""
    apply_coupling!(system, coupling, simulator, neighbors=nothing,
                    step_n=0; n_threads=Threads.nthreads())

Apply a coupler to modify a simulation.

Returns whether the coupling has invalidated the currently stored forces, for
example by changing the coordinates.
This information is useful for some simulators.
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

The Andersen thermostat for controlling temperature.

The velocity of each atom is randomly changed each time step with probability
`dt / coupling_const` to a velocity drawn from the Maxwell-Boltzmann distribution.
See [Andersen 1980](https://doi.org/10.1063/1.439486).
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

The velocity rescaling thermostat for controlling temperature.

Velocities are immediately rescaled to match a target temperature.
The scaling factor for the velocities each step is
```math
\lambda = \sqrt{\frac{T_0}{T}}
```

This thermostat should be used with caution as it can lead to simulation
artifacts.
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

The Berendsen thermostat for controlling temperature.

The scaling factor for the velocities each step is
```math
\lambda^2 = 1 + \frac{\delta t}{\tau} \left( \frac{T_0}{T} - 1 \right)
```

This thermostat should be used with caution as it can lead to simulation
artifacts.
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

@doc raw"""
    MonteCarloBarostat(pressure, temperature, boundary; n_steps=30, n_iterations=1,
                       scale_factor=0.01, scale_increment=1.1, max_volume_frac=0.3,
                       trial_find_neighbors=false)

The Monte Carlo barostat for controlling pressure.

See [Chow and Ferguson 1995](https://doi.org/10.1016/0010-4655(95)00059-O),
[Åqvist et al. 2004](https://doi.org/10.1016/j.cplett.2003.12.039) and the OpenMM
source code.
At regular intervals a Monte Carlo step is attempted by scaling the coordinates and
the bounding box by a randomly chosen amount.
The step is accepted or rejected based on
```math
\Delta W = \Delta E + P \Delta V - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `P` is the equilibrium pressure, `ΔV` is
the change in volume, `N` is the number of molecules in the system, `T` is the equilibrium
temperature and `V` is the system volume.
If `ΔW ≤ 0` the step is always accepted, if `ΔW > 0` the step is accepted with probability
`exp(-ΔW/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system.

Not currently compatible with automatic differentiation using Zygote.
"""
mutable struct MonteCarloBarostat{T, P, K, V}
    pressure::P
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Int
    n_accepted::Int
end

function MonteCarloBarostat(P, T, boundary; n_steps=30, n_iterations=1, scale_factor=0.01,
                            scale_increment=1.1, max_volume_frac=0.3, trial_find_neighbors=false)
    volume_scale = box_volume(boundary) * float_type(boundary)(scale_factor)
    return MonteCarloBarostat(P, T, n_steps, n_iterations, volume_scale, scale_increment,
                              max_volume_frac, trial_find_neighbors, 0, 0)
end

function apply_coupling!(sys::System{D, G, T}, barostat::MonteCarloBarostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads()) where {D, G, T}
    if !iszero(step_n % barostat.n_steps)
        return false
    end

    kT = sys.k * barostat.temperature
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false

    for attempt_n in 1:barostat.n_iterations
        E = potential_energy(sys, neighbors; n_threads=n_threads)
        V = box_volume(sys.boundary)
        dV = barostat.volume_scale * (2 * rand(T) - 1)
        v_scale = (V + dV) / V
        l_scale = (D == 2 ? sqrt(v_scale) : cbrt(v_scale))
        old_coords = copy(sys.coords)
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

        if barostat.trial_find_neighbors
            neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                             n_threads=n_threads)
        else
            # Assume neighbors are unchanged by the change in coordinates
            # This may not be valid for larger changes
            neighbors_trial = neighbors
        end
        E_trial = potential_energy(sys, neighbors_trial; n_threads=n_threads)
        dE = energy_remove_mol(E_trial - E)
        dW = dE + uconvert(unit(dE), barostat.pressure * dV) - n_molecules * kT * log(v_scale)
        if dW <= zero(dW) || rand(T) < exp(-dW / kT)
            recompute_forces = true
            barostat.n_accepted += 1
        else
            sys.coords = old_coords
            sys.boundary = old_boundary
        end
        barostat.n_attempted += 1

        # Modify size of volume change to keep accept/reject ratio roughly equal
        if barostat.n_attempted >= 10
            if barostat.n_accepted < 0.25 * barostat.n_attempted
                barostat.volume_scale /= barostat.scale_increment
                barostat.n_attempted = 0
                barostat.n_accepted = 0
            elseif barostat.n_accepted > 0.75 * barostat.n_attempted
                barostat.volume_scale = min(barostat.volume_scale * barostat.scale_increment,
                                            V * barostat.max_volume_frac)
                barostat.n_attempted = 0
                barostat.n_accepted = 0
            end
        end
    end
    return recompute_forces
end

@doc raw"""
    MonteCarloAnisotropicBarostat(pressure_X, pressure_Y, pressure_Z, temperature, boundary;
                       n_steps=30, n_iterations=1, scale_factor=0.01, scale_increment=1.1,
                       max_volume_frac=0.3, trial_find_neighbors=false)

The Monte Carlo anisotropic barostat for controlling pressure.

See [Chow and Ferguson 1995](https://doi.org/10.1016/0010-4655(95)00059-O),
[Åqvist et al. 2004](https://doi.org/10.1016/j.cplett.2003.12.039) and the OpenMM
source code.
At regular intervals a Monte Carlo step is attempted by scaling the coordinates and
the bounding box by a randomly chosen amount in a randomly selected axis.
The step is accepted or rejected based on
```math
\Delta W = \Delta E + P \Delta V - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `P` is the equilibrium pressure along the
selected axis, `ΔV` is the change in volume, `N` is the number of molecules in the system,
`T` is the equilibrium temperature and `V` is the system volume.
If `ΔW ≤ 0` the step is always accepted, if `ΔW > 0` the step is accepted with probability
`exp(-ΔW/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system.
To keep an axis fixed, set the corresponding pressure to `nothing`.
For rectangular boundaries `pressure_Z` is not required.

Not currently compatible with automatic differentiation using Zygote.
"""
mutable struct MonteCarloAnisotropicBarostat{T, P, K, V}
    pressure::P
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Vector{Int}
    n_accepted::Vector{Int}
end

function MonteCarloAnisotropicBarostat(
    pressure_X,
    pressure_Y,
    pressure_Z,
    temperature,
    boundary;
    n_steps=30,
    n_iterations=1,
    scale_factor=0.01,
    scale_increment=1.1,
    max_volume_frac=0.3,
    trial_find_neighbors=false,
)
    pressure = SVector(pressure_X, pressure_Y, pressure_Z)
    volume_scale = box_volume(boundary) * float_type(boundary)(scale_factor)
    volume_scale = fill(volume_scale, 3)

    return MonteCarloAnisotropicBarostat(
        pressure,
        temperature,
        n_steps,
        n_iterations,
        volume_scale,
        scale_increment,
        max_volume_frac,
        trial_find_neighbors,
        fill(0, 3),
        fill(0, 3),
    )
end

function MonteCarloAnisotropicBarostat(
    pressure_X,
    pressure_Y,
    temperature,
    boundary::RectangularBoundary;
    n_steps=30,
    n_iterations=1,
    scale_factor=0.01,
    scale_increment=1.1,
    max_volume_frac=0.3,
    trial_find_neighbors=false,
)
    pressure = SVector(pressure_X, pressure_Y)
    volume_scale = box_volume(boundary) * float_type(boundary)(scale_factor)
    volume_scale = fill(volume_scale, 2)

    return MonteCarloAnisotropicBarostat(
        pressure,
        temperature,
        n_steps,
        n_iterations,
        volume_scale,
        scale_increment,
        max_volume_frac,
        trial_find_neighbors,
        fill(0, 2),
        fill(0, 2),
    )
end

function apply_coupling!(
    sys::System{D, G, T},
    barostat::MonteCarloAnisotropicBarostat,
    sim,
    neighbors=nothing,
    step_n::Integer=0;
    n_threads::Integer=Threads.nthreads()
) where {D, G, T}

    !iszero(step_n % barostat.n_steps) && return false
    all(isnothing, barostat.pressure) && return false

    kT = sys.k * barostat.temperature
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false

    axis = undef
    while true
        axis = rand(1:D)
        !isnothing(barostat.pressure[axis]) && break
    end
    mask1 = falses(D)
    mask2 = trues(D)
    mask1[axis] = true
    mask2[axis] = false

    for attempt_n in 1:barostat.n_iterations
        E = potential_energy(sys, neighbors; n_threads=n_threads)
        V = box_volume(sys.boundary)
        dV = barostat.volume_scale[axis] * (2 * rand(T) - 1)
        v_scale = (V + dV) / V
        l_scale = SVector{D}(mask1 * (D == 2 ? sqrt(v_scale) : cbrt(v_scale)) + mask2)
        old_coords = copy(sys.coords)
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

        if barostat.trial_find_neighbors
            neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                             n_threads=n_threads)
        else
            # Assume neighbors are unchanged by the change in coordinates
            # This may not be valid for larger changes
            neighbors_trial = neighbors
        end
        E_trial = potential_energy(sys, neighbors_trial; n_threads=n_threads)
        dE = energy_remove_mol(E_trial - E)
        dW = dE + uconvert(unit(dE), barostat.pressure[axis] * dV) - n_molecules * kT * log(v_scale)
        if dW <= zero(dW) || rand(T) < exp(-dW / kT)
            recompute_forces = true
            barostat.n_accepted[axis] += 1
        else
            sys.coords = old_coords
            sys.boundary = old_boundary
        end
        barostat.n_attempted[axis] += 1

        # Modify size of volume change to keep accept/reject ratio roughly equal
        if barostat.n_attempted[axis] >= 10
          if barostat.n_accepted[axis] < 0.25 * barostat.n_attempted[axis]
            barostat.volume_scale[axis] /= barostat.scale_increment
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
              elseif barostat.n_accepted[axis] > 0.75 * barostat.n_attempted[axis]
                barostat.volume_scale[axis] = min(barostat.volume_scale[axis] * barostat.scale_increment,
                                            V * barostat.max_volume_frac)
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
            end
        end
    end
    return recompute_forces
end
