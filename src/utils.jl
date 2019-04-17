# General utils - neighbour finders, thermostats etc

export
    DistanceNeighbourFinder,
    find_neighbours!,
    AndersenThermostat,
    apply_thermostat!,
    maxwellboltzmann

"Find close atoms by distance."
struct DistanceNeighbourFinder <: NeighbourFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    sqdist_cutoff::Float64
end

function DistanceNeighbourFinder(nb_matrix::BitArray{2},
                                n_steps::Integer)
    return DistanceNeighbourFinder(nb_matrix, n_steps, 1.2 ^ 2)
end

"Update list of close atoms between which non-bonded forces are calculated."
function find_neighbours!(s::Simulation, nf::DistanceNeighbourFinder)
    empty!(s.neighbour_list)
    for i in 1:length(s.coords)
        ci = s.coords[i]
        nbi = s.neighbour_finder.nb_matrix[:, i]
        for j in 1:(i - 1)
            if sqdist(ci, s.coords[j], s.box_size) <= nf.sqdist_cutoff && nbi[j]
                push!(s.neighbour_list, (i, j))
            end
        end
    end
    return s
end

"Rescale random velocities according to the Andersen thermostat."
struct AndersenThermostat <: Thermostat
    coupling_const::Float64
end

"Apply a thermostat to modify a simulation."
function apply_thermostat!(s::Simulation, thermostat::AndersenThermostat)
    for (i, v) in enumerate(s.velocities)
        if rand() < s.timestep / thermostat.coupling_const
            mass = s.atoms[i].mass
            v.x = maxwellboltzmann(mass, s.temperature)
            v.y = maxwellboltzmann(mass, s.temperature)
            v.z = maxwellboltzmann(mass, s.temperature)
        end
    end
end

"Generate a random velocity from the Maxwell-Boltzmann distribution."
function maxwellboltzmann(mass::Real, T::Real)
    return rand(Normal(0.0, sqrt(molar_gas_const * T / mass)))
end

"Calculate the temperature of a system from the kinetic energy of the atoms."
function temperature(s::Simulation)
    ke = sum([a.mass * dot(s.velocities[i], s.velocities[i]) for (i, a) in enumerate(s.atoms)]) / 2
    df = 3 * length(s.coords) - 3
    return 2 * ke / df
end

# Placeholder if we want to change messaging later on
report(msg...) = println(msg...)
