# Different ways to simulate molecules

export
    calc_accelerations,
    VelocityVerlet,
    simulate!

function adjust_bounds(c::Real, box_size::Real)
    while c >= box_size
        c -= box_size
    end
    while c < 0.0
        c += box_size
    end
    return c
end

"Calculate accelerations of all atoms using the bonded and non-bonded forces."
function calc_accelerations(s::Simulation; parallel::Bool=true)
    n_atoms = length(s.coords)

    if parallel && nthreads() > 1 && n_atoms > 100
        accels_threads = [zero(s.coords) for i in 1:nthreads()]

        # Loop over interactions and calculate the acceleration due to each
        for inter in values(s.general_inters)
            if inter.nl_only
                @threads for ni in 1:length(s.neighbour_list)
                    i, j = s.neighbour_list[ni]
                    update_accelerations!(accels_threads[threadid()], inter, s, i, j)
                end
            else
                @threads for i in 1:n_atoms
                    for j in 1:(i - 1)
                        update_accelerations!(accels_threads[threadid()], inter, s, i, j)
                    end
                end
            end
        end

        accels = sum(accels_threads)
    else
        accels = zero(s.coords)

        for inter in values(s.general_inters)
            if inter.nl_only
                for ni in 1:length(s.neighbour_list)
                    i, j = s.neighbour_list[ni]
                    update_accelerations!(accels, inter, s, i, j)
                end
            else
                for i in 1:n_atoms
                    for j in 1:(i - 1)
                        update_accelerations!(accels, inter, s, i, j)
                    end
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        for inter in inter_list
            update_accelerations!(accels, inter, s)
        end
    end

    # Divide sum of forces by the atom mass to get acceleration
    for i in 1:n_atoms
        accels[i] /= s.atoms[i].mass
    end

    return accels
end

"The velocity Verlet integrator."
struct VelocityVerlet <: Simulator end

# See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
#   integration algorithm - used shorter second version
"Simulate molecular dynamics."
function simulate!(s::Simulation,
                    ::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    n_atoms = length(s.coords)
    find_neighbours!(s, s.neighbour_finder, 0, parallel=parallel)
    a_t = calc_accelerations(s, parallel=parallel)
    a_t_dt = zero(s.coords)

    @showprogress for step_n in 1:n_steps
        # Update coordinates
        for i in 1:length(s.coords)
            s.coords[i] += s.velocities[i] * s.timestep + 0.5 * a_t[i] * s.timestep ^ 2
            s.coords[i] = adjust_bounds.(s.coords[i], s.box_size)
        end

        a_t_dt = calc_accelerations(s, parallel=parallel)

        # Update velocities
        for i in 1:length(s.velocities)
            s.velocities[i] += 0.5 * (a_t[i] + a_t_dt[i]) * s.timestep
        end

        apply_thermostat!(s, s.thermostat)
        find_neighbours!(s, s.neighbour_finder, step_n, parallel=parallel)
        for logger in values(s.loggers)
            log_property!(logger, s, step_n)
        end

        a_t = a_t_dt
        s.n_steps_made[1] += 1
    end
    return s
end

function simulate!(s::Simulation, n_steps::Integer; parallel::Bool=true)
    simulate!(s, s.simulator, n_steps, parallel=parallel)
end

function simulate!(s::Simulation; parallel::Bool=true)
    simulate!(s, s.n_steps - first(s.n_steps_made), parallel=parallel)
end
