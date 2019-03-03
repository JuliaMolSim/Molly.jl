# Different ways to simulate molecules

export
    Simulator,
    Simulation,
    update_coordinates!,
    update_velocities!,
    calc_accelerations!,
    empty_accelerations,
    VelocityVerlet,
    simulate!

"A type of simulation to run, e.g. leap-frog integration or energy minimisation."
abstract type Simulator end

"The data associated with a molecular simulation."
mutable struct Simulation
    simulator::Simulator
    atoms::Vector{Atom}
    specific_inter_lists::Dict{String, Vector{T} where T <: SpecificInteraction}
    general_inters::Dict{String, GeneralInteraction}
    coords::Vector{Coordinates}
    velocities::Vector{Velocity}
    temperature::Float64
    box_size::Float64
    neighbour_list::Vector{Tuple{Int, Int}}
    neighbour_finder::NeighbourFinder
    thermostat::Thermostat
    loggers::Vector{Logger}
    timestep::Float64
    n_steps::Int
    n_steps_made::Int
end

"Update coordinates of all atoms and bound to the bounding box."
function update_coordinates!(s::Simulation, accels::Vector{Acceleration})
    for (i, c) in enumerate(s.coords)
        c.x += s.velocities[i].x * s.timestep + 0.5 * accels[i].x * s.timestep ^ 2
        while (c.x >= s.box_size) c.x -= s.box_size end
        while (c.x < 0.0) c.x += s.box_size end

        c.y += s.velocities[i].y * s.timestep + 0.5 * accels[i].y * s.timestep ^ 2
        while (c.y >= s.box_size) c.y -= s.box_size end
        while (c.y < 0.0) c.y += s.box_size end

        c.z += s.velocities[i].z * s.timestep + 0.5 * accels[i].z * s.timestep ^ 2
        while (c.z >= s.box_size) c.z -= s.box_size end
        while (c.z < 0.0) c.z += s.box_size end
    end
    return s.coords
end

"Update velocities of all atoms using the accelerations."
function update_velocities!(s::Simulation,
                    accels_t::Vector{Acceleration},
                    accels_t_dt::Vector{Acceleration})
    for (i, v) in enumerate(s.velocities)
        v.x += 0.5 * (accels_t[i].x + accels_t_dt[i].x) * s.timestep
        v.y += 0.5 * (accels_t[i].y + accels_t_dt[i].y) * s.timestep
        v.z += 0.5 * (accels_t[i].z + accels_t_dt[i].z) * s.timestep
    end
    return s.velocities
end

"Calculate accelerations of all atoms using the bonded and non-bonded forces."
function calc_accelerations!(accels::Vector{Acceleration}, s::Simulation)
    # Empty accelerations
    for a in accels
        fill!(a, 0.0)
    end

    # Loop over interactions and calculate the acceleration due to each
    n_atoms = length(s.coords)
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
        accels[i].x /= s.atoms[i].mass
        accels[i].y /= s.atoms[i].mass
        accels[i].z /= s.atoms[i].mass
    end

    return accels
end

"Initialise empty `Acceleration`s."
empty_accelerations(n_atoms::Integer) = [Acceleration(0.0, 0.0, 0.0) for i in 1:n_atoms]

"The velocity Verlet integrator."
struct VelocityVerlet <: Simulator end

# See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
#   integration algorithm - used shorter second version
"Simulate molecular dynamics."
function simulate!(s::Simulation, ::VelocityVerlet, n_steps::Integer)
    n_atoms = length(s.coords)
    find_neighbours!(s, s.neighbour_finder)
    a_t = calc_accelerations!(empty_accelerations(n_atoms), s)
    a_t_dt = empty_accelerations(n_atoms)
    @showprogress for step_n in 1:n_steps
        update_coordinates!(s, a_t)
        calc_accelerations!(a_t_dt, s)
        update_velocities!(s, a_t, a_t_dt)
        apply_thermostat!(s, s.thermostat)
        if step_n % s.neighbour_finder.n_steps == 0
            find_neighbours!(s, s.neighbour_finder)
        end
        for logger in s.loggers
            log_property!(logger, s, step_n)
        end
        a_t = a_t_dt
        s.n_steps_made += 1
    end
end

function simulate!(s::Simulation, n_steps::Integer)
    report("Starting simulation")
    simulate!(s, s.simulator, n_steps)
    report("Simulation finished")
    return s
end

simulate!(s::Simulation) = simulate!(s, s.n_steps - s.n_steps_made)
