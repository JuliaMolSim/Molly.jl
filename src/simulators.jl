# Different ways to simulate molecules

export
    VelocityVerlet,
    StormerVerlet,
    simulate!

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
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
end

VelocityVerlet(; dt, coupling=NoCoupling()) = VelocityVerlet(dt, coupling)

"""
    simulate!(system, simulator, n_steps; parallel=true)

Run a simulation on a system according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(sys::System{D, false},
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true) where D
    # See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
    #   integration algorithm - used shorter second version
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    accels_t = accelerations(sys, neighbors; parallel=parallel)
    accels_t_dt = zero(accels_t)

    @showprogress for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        # Update coordinates
        for i in 1:length(sys)
            sys.coords[i] += sys.velocities[i] * sim.dt + remove_molar(accels_t[i]) * (sim.dt ^ 2) / 2
            sys.coords[i] = wrap_coords.(sys.coords[i], sys.box_size)
        end

        accels_t_dt = accelerations(sys, neighbors; parallel=parallel)

        # Update velocities
        for i in 1:length(sys)
            sys.velocities[i] += remove_molar(accels_t[i] + accels_t_dt[i]) * sim.dt / 2
        end

        apply_coupling!(sys, sim, sim.coupling)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; parallel=parallel)
            accels_t = accels_t_dt
        end
    end
    return sys
end

function simulate!(sys::System{D, true},
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true) where D
    neighbors = find_neighbors(sys, sys.neighbor_finder)
    accels_t = accelerations(sys, neighbors)
    accels_t_dt = zero(accels_t)

    for step_n in 1:n_steps
        run_loggers!(sys, neighbors, step_n)

        sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))
        accels_t_dt = accelerations(sys, neighbors)
        sys.velocities += remove_molar.(accels_t .+ accels_t_dt) .* sim.dt / 2

        apply_coupling!(sys, sim, sim.coupling)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n)
            accels_t = accels_t_dt
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

function simulate!(sys::System,
                    sim::StormerVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)

    @showprogress for step_n in 1:n_steps
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
