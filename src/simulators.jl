# Different ways to simulate molecules

export
    simulate!,
    VelocityVerlet,
    VelocityFreeVerlet

# Forces are often expressed per mol but this dimension needs removing for use in the integrator
function removemolar(x)
    fx = first(x)
    if dimension(fx) == u"𝐋 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(fx))
        return x / T(Unitful.Na)
    else
        return x
    end
end

"""
    VelocityVerlet()

The velocity Verlet integrator.
"""
struct VelocityVerlet end

"""
    simulate!(system; parallel=true)
    simulate!(system, n_steps; parallel=true)
    simulate!(system, simulator, n_steps; parallel=true)

Run a simulation on a system according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(s::System{D, S, false},
                    ::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true) where {D, S}
    # See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
    #   integration algorithm - used shorter second version
    neighbors = find_neighbors(s, s.neighbor_finder; parallel=parallel)
    accels_t = accelerations(s, neighbors; parallel=parallel)
    accels_t_dt = zero(accels_t)

    @showprogress for step_n in 1:n_steps
        run_loggers!(s, neighbors, step_n)

        # Update coordinates
        for i in 1:length(s)
            s.coords[i] += s.velocities[i] * s.timestep + removemolar(accels_t[i]) * (s.timestep ^ 2) / 2
            s.coords[i] = wrapcoords.(s.coords[i], s.box_size)
        end

        accels_t_dt = accelerations(s, neighbors; parallel=parallel)

        # Update velocities
        for i in 1:length(s)
            s.velocities[i] += removemolar(accels_t[i] + accels_t_dt[i]) * s.timestep / 2
        end

        apply_coupling!(s, s.coupling)
        neighbors = find_neighbors(s, s.neighbor_finder, neighbors, step_n; parallel=parallel)

        accels_t = accels_t_dt
    end
    return s
end

function simulate!(s::System{D, S, true},
                    ::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true) where {D, S}
    if length([inter for inter in values(s.general_inters) if !inter.nl_only]) > 0
        neighbors_all = allneighbors(length(s))
    else
        neighbors_all = nothing
    end
    neighbors = find_neighbors(s, s.neighbor_finder)
    accels_t = accelerations(s, s.coords, s.atoms, neighbors, neighbors_all)
    accels_t_dt = zero(accels_t)

    for step_n in 1:n_steps
        run_loggers!(s, neighbors, step_n)

        s.coords += s.velocities .* s.timestep .+ (removemolar.(accels_t) .* s.timestep ^ 2) ./ 2
        s.coords = wrapcoordsvec.(s.coords, (s.box_size,))
        accels_t_dt = accelerations(s, s.coords, s.atoms, neighbors, neighbors_all)
        s.velocities += removemolar.(accels_t .+ accels_t_dt) .* s.timestep / 2

        apply_coupling!(s, s.coupling)
        neighbors = find_neighbors(s, s.neighbor_finder, neighbors, step_n)

        accels_t = accels_t_dt
    end
    return s
end

"""
    VelocityFreeVerlet()

The velocity-free Verlet integrator, also known as the Störmer method.
In this case the `velocities` given to the simulator act as the previous step
coordinates for the first step.
"""
struct VelocityFreeVerlet end

function simulate!(s::System,
                    ::VelocityFreeVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(s, s.neighbor_finder; parallel=parallel)

    @showprogress for step_n in 1:n_steps
        run_loggers!(s, neighbors, step_n)

        accels_t = accelerations(s, neighbors; parallel=parallel)

        # Update coordinates
        coords_copy = s.coords
        for i in 1:length(s)
            s.coords[i] = s.coords[i] + vector(s.velocities[i], s.coords[i], s.box_size) + removemolar(accels_t[i]) * s.timestep ^ 2
            s.coords[i] = wrapcoords.(s.coords[i], s.box_size)
        end
        s.velocities = coords_copy

        apply_coupling!(s, s.coupling)
        neighbors = find_neighbors(s, s.neighbor_finder, neighbors, step_n; parallel=parallel)
    end
    return s
end

function simulate!(s::System, n_steps::Integer; parallel::Bool=true)
    simulate!(s, s.simulator, n_steps; parallel=parallel)
end

function simulate!(s::System; parallel::Bool=true)
    simulate!(s, s.n_steps; parallel=parallel)
end
