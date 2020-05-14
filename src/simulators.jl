# Different ways to simulate molecules

export
    update_coordinates!,
    update_velocities!,
    calc_accelerations,
    empty_accelerations,
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

"Update coordinates of all atoms and bound to the bounding box."
function update_coordinates!(s::Simulation, accels::Vector{T}) where T
    for i in 1:length(s.coords)
        s.coords[i] += s.velocities[i] * s.timestep + 0.5 * accels[i] * s.timestep ^ 2
        s.coords[i] = adjust_bounds.(s.coords[i], s.box_size)
    end
    return s.coords
end

"Update velocities of all atoms using the accelerations."
function update_velocities!(s::Simulation,
                    accels_t::Vector{T},
                    accels_t_dt::Vector{T}) where T
    for i in 1:length(s.velocities)
        s.velocities[i] += 0.5 * (accels_t[i] + accels_t_dt[i]) * s.timestep
    end
    return s.velocities
end

"Calculate accelerations of all atoms using the bonded and non-bonded forces."
function calc_accelerations(s::Simulation)
    n_atoms = length(s.coords)
    accels = empty_accelerations(n_atoms)

    # Loop over interactions and calculate the acceleration due to each
    for inter_list in values(s.specific_inter_lists)
        for inter in inter_list
            update_accelerations!(accels, inter, s)
        end
    end

    for inter in values(s.general_inters)
        if inter.nl_only
            for (i, j) in s.neighbour_list
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

    # Divide sum of forces by the atom mass to get acceleration
    for i in 1:n_atoms
        accels[i] /= s.atoms[i].mass
    end

    return accels
end

"Initialise empty `Acceleration`s."
empty_accelerations(n_atoms::Integer) = [SVector(0.0, 0.0, 0.0) for i in 1:n_atoms]

"The velocity Verlet integrator."
struct VelocityVerlet <: Simulator end

# See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
#   integration algorithm - used shorter second version
"Simulate molecular dynamics."
function simulate!(s::Simulation, ::VelocityVerlet, n_steps::Integer)
    n_atoms = length(s.coords)
    find_neighbours!(s, s.neighbour_finder, 0)
    a_t = calc_accelerations(s)
    a_t_dt = empty_accelerations(n_atoms)
    @showprogress for step_n in 1:n_steps
        update_coordinates!(s, a_t)
        a_t_dt = calc_accelerations(s)
        update_velocities!(s, a_t, a_t_dt)
        apply_thermostat!(s, s.thermostat)
        find_neighbours!(s, s.neighbour_finder, step_n)
        for logger in s.loggers
            log_property!(logger, s, step_n)
        end
        a_t = a_t_dt
        s.n_steps_made += 1
    end
    return s
end

function simulate!(s::Simulation, n_steps::Integer)
    simulate!(s, s.simulator, n_steps)
    return s
end

simulate!(s::Simulation) = simulate!(s, s.n_steps - s.n_steps_made)
