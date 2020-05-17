# General utils - neighbour finders, thermostats etc

export
    DistanceNeighbourFinder,
    find_neighbours!,
    NoNeighbourFinder,
    AndersenThermostat,
    apply_thermostat!,
    NoThermostat,
    velocity,
    maxwellboltzmann,
    temperature

"Find close atoms by distance."
struct DistanceNeighbourFinder <: NeighbourFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::Float64
end

function DistanceNeighbourFinder(nb_matrix::BitArray{2},
                                n_steps::Integer)
    return DistanceNeighbourFinder(nb_matrix, n_steps, 1.2)
end

"Update list of close atoms between which non-bonded forces are calculated."
function find_neighbours!(s::Simulation,
                            nf::DistanceNeighbourFinder,
                            step_n::Integer;
                            parallel::Bool=true)
    if step_n % nf.n_steps == 0
        empty!(s.neighbour_list)
        sqdist_cutoff = nf.dist_cutoff ^ 2

        if parallel && nthreads() > 1
            nl_threads = [Tuple{Int, Int}[] for i in 1:nthreads()]

            @threads for i in 1:length(s.coords)
                nl = nl_threads[threadid()]
                ci = s.coords[i]
                nbi = nf.nb_matrix[:, i]
                for j in 1:(i - 1)
                    r2 = sum(abs2, vector1D.(ci, s.coords[j], s.box_size))
                    if r2 <= sqdist_cutoff && nbi[j]
                        push!(nl, (i, j))
                    end
                end
            end

            for nl in nl_threads
                append!(s.neighbour_list, nl)
            end
        else
            for i in 1:length(s.coords)
                ci = s.coords[i]
                nbi = nf.nb_matrix[:, i]
                for j in 1:(i - 1)
                    r2 = sum(abs2, vector1D.(ci, s.coords[j], s.box_size))
                    if r2 <= sqdist_cutoff && nbi[j]
                        push!(s.neighbour_list, (i, j))
                    end
                end
            end
        end
    end
    return s
end

"Placeholder neighbour finder that does nothing."
struct NoNeighbourFinder <: NeighbourFinder end

function find_neighbours!(s::Simulation, ::NoNeighbourFinder, ::Integer; kwargs...)
    return s
end

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
