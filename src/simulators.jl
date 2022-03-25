# Different ways to simulate molecules

export
    VelocityVerlet,
    simulate!,
    Verlet,
    StormerVerlet,
    Langevin

# Forces are often expressed per mol but this dimension needs removing for use in the integrator
function remove_molar(x)
    fx = first(x)
    if dimension(fx) == u"𝐋 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(fx))
        return x / T(Unitful.Na)
    else
        return x
    end
end

"""
    VelocityVerlet(; <keyword arguments>)

The velocity Verlet integrator.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function VelocityVerlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return VelocityVerlet(dt, coupling, remove_CM_motion)
end

"""
    simulate!(system, simulator, n_steps; parallel=true)

Run a simulation on a system according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(sys,
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    accels_t = accelerations(sys, neighbors; parallel=parallel)
    accels_t_dt = zero(accels_t)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))
        accels_t_dt = accelerations(sys, neighbors; parallel=parallel)
        sys.velocities += remove_molar.(accels_t .+ accels_t_dt) .* sim.dt / 2

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim, sim.coupling)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
            accels_t = accels_t_dt
        end
    end
    return sys
end

"""
    Verlet(; <keyword arguments>)

The leapfrog Verlet integrator.
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct Verlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function Verlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return Verlet(dt, coupling, remove_CM_motion)
end

function simulate!(sys,
                    sim::Verlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        accels_t = accelerations(sys, neighbors; parallel=parallel)
        sys.velocities += remove_molar.(accels_t) .* sim.dt

        sys.coords += sys.velocities .* sim.dt
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim, sim.coupling)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
        end
    end
    return sys
end

"""
    StormerVerlet(; <keyword arguments>)

The Störmer-Verlet integrator.
In this case the `velocities` given to the simulator act as the previous step
coordinates for the first step.
Does not currently work with units or thermostats.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
"""
struct StormerVerlet{T, C}
    dt::T
    coupling::C
end

StormerVerlet(; dt, coupling=NoCoupling()) = StormerVerlet(dt, coupling)

function simulate!(sys,
                    sim::StormerVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)

    for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        accels_t = accelerations(sys, neighbors; parallel=parallel)

        # Update coordinates
        coords_copy = sys.coords
        for i in 1:length(sys)
            sys.coords[i] = sys.coords[i] + vector(sys.velocities[i], sys.coords[i], sys.box_size) + remove_molar(accels_t[i]) * sim.dt ^ 2
            sys.coords[i] = wrap_coords.(sys.coords[i], sys.box_size)
        end
        sys.velocities = coords_copy

        apply_coupling!(sys, sim, sim.coupling)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; parallel=parallel)
        end
    end
    return sys
end

"""
    Langevin(; <keyword arguments>)

The Langevin integrator, based on the Langevin Middle Integrator in OpenMM.
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `temperature::K`: the temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct Langevin{S, K, F, T}
    dt::S
    temperature::K
    friction::F
    remove_CM_motion::Bool
    vel_scale::T
    noise_scale::T
end

function Langevin(; dt, temperature, friction, remove_CM_motion=true)
    vel_scale = exp(-dt * friction)
    noise_scale = sqrt(1 - vel_scale^2)
    return Langevin(dt, temperature, friction, remove_CM_motion, vel_scale, noise_scale)
end

function simulate!(sys,
                    sim::Langevin,
                    n_steps::Integer;
                    parallel::Bool=true,
                    rng=Random.GLOBAL_RNG)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        accels_t = accelerations(sys, neighbors; parallel=parallel)
        sys.velocities += remove_molar.(accels_t) .* sim.dt

        sys.coords += sys.velocities .* sim.dt / 2
        noise = random_velocities(sys, sim.temperature; rng=rng)
        sys.velocities = sys.velocities .* sim.vel_scale .+ noise .* sim.noise_scale

        sys.coords += sys.velocities .* sim.dt / 2
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        sim.remove_CM_motion && remove_CM_motion!(sys)
        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
        end
    end
    return sys
end
