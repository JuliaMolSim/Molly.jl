using Molly
using Test

temperature = 298
timestep = 0.002
n_steps = 1_000
box_size = 2.0

@testset "Lennard-Jones gas 2D" begin
    n_atoms = 20

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                    mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        general_inters=(LennardJones(true),),
        coords=[box_size .* rand(SVector{2}) for i in 1:n_atoms],
        velocities=[velocity(10.0, temperature, dims=2) .* 0.01 for i in 1:n_atoms],
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(100),
                        "coords" => CoordinateLogger(100, dims=2)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@testset "Lennard-Jones gas" begin
    n_atoms = 100

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                    mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        general_inters=(LennardJones(true),),
        coords=[box_size .* rand(SVector{3}) for i in 1:n_atoms],
        velocities=[velocity(10.0, temperature) .* 0.01 for i in 1:n_atoms],
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(100),
                        "coords" => CoordinateLogger(100)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)

    final_coords = s.loggers["coords"].coords[end]
    displacements(final_coords, box_size)
    distances(final_coords, box_size)
    rdf(final_coords, box_size)
end

@testset "Lennard-Jones gas velocity-free" begin
    n_atoms = 100
    coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]

    s = Simulation(
        simulator=VelocityFreeVerlet(),
        atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                    mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        general_inters=(LennardJones(true),),
        coords=coords,
        velocities=[c .+ 0.01 .* rand(SVector{3}) for c in coords],
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(100),
                        "coords" => CoordinateLogger(100)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@testset "Diatomic molecules" begin
    n_atoms = 100
    coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]
    for i in 1:length(coords)
        push!(coords, coords[i] .+ [0.1, 0.0, 0.0])
    end
    bonds = [HarmonicBond(i, Int(i + n_atoms / 2), 0.1, 300_000.0) for i in 1:Int(n_atoms / 2)]

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=[Atom(attype="H", name="H", resnum=i, resname="H", charge=0.0,
                    mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        specific_inter_lists=(bonds,),
        general_inters=(LennardJones(true),),
        coords=coords,
        velocities=[velocity(10.0, temperature) .* 0.01 for i in 1:n_atoms],
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(10),
                        "coords" => CoordinateLogger(10)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@testset "Peptide" begin
    timestep = 0.0002
    n_steps = 100
    atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
                normpath(@__DIR__, "..", "data", "5XER", "gmx_top_ff.top"),
                normpath(@__DIR__, "..", "data", "5XER", "gmx_coords.gro"))

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=[velocity(a.mass, temperature) .* 0.01 for a in atoms],
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(10)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@testset "Float32" begin
    timestep = 0.0002f0
    n_steps = 100
    atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
                Float32,
                normpath(@__DIR__, "..", "data", "5XER", "gmx_top_ff.top"),
                normpath(@__DIR__, "..", "data", "5XER", "gmx_coords.gro"))

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=[velocity(Float32, a.mass, temperature) .* 0.01f0 for a in atoms],
        temperature=Float32(temperature),
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10, 1.2f0),
        thermostat=AndersenThermostat(10.0f0),
        loggers=Dict("temp" => TemperatureLogger(10)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@enum Status susceptible infected recovered

# Custom atom type
mutable struct Person
    status::Status
    mass::Float64
    σ::Float64
    ϵ::Float64
end

# Custom GeneralInteraction
struct SIRInteraction <: GeneralInteraction
    nl_only::Bool
    dist_infection::Float64
    prob_infection::Float64
    prob_recovery::Float64
end

# Custom Logger
struct SIRLogger <: Logger
    n_steps::Int
    fracs_sir::Vector{Vector{Float64}}
end

@testset "Agent-based modelling" begin
    # Custom force function
    function Molly.force!(forces, inter::SIRInteraction, s::Simulation, i::Integer, j::Integer)
        if i == j
            # Recover randomly
            if s.atoms[i].status == infected && rand() < inter.prob_recovery
                s.atoms[i].status = recovered
            end
        elseif (s.atoms[i].status == infected && s.atoms[j].status == susceptible) ||
                    (s.atoms[i].status == susceptible && s.atoms[j].status == infected)
            # Infect close people randomly
            dr = vector(s.coords[i], s.coords[j], s.box_size)
            r2 = sum(abs2, dr)
            if r2 < inter.dist_infection ^ 2 && rand() < inter.prob_infection
                s.atoms[i].status = infected
                s.atoms[j].status = infected
            end
        end
        return forces
    end

    # Custom logging function
    function Molly.log_property!(logger::SIRLogger, s::Simulation, step_n::Integer)
        if step_n % logger.n_steps == 0
            counts_sir = [
                count(p -> p.status == susceptible, s.atoms),
                count(p -> p.status == infected   , s.atoms),
                count(p -> p.status == recovered  , s.atoms)
            ]
            push!(logger.fracs_sir, counts_sir ./ length(s.atoms))
        end
    end

    temperature = 0.01
    timestep = 0.02
    box_size = 10.0
    n_steps = 1_000
    n_people = 500
    n_starting = 2
    atoms = [Person(i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]
    coords = [box_size .* rand(SVector{2}) for i in 1:n_people]
    velocities = [velocity(1.0, temperature, dims=2) for i in 1:n_people]
    general_inters = Dict("LennardJones" => LennardJones(true),
                            "SIR" => SIRInteraction(false, 0.5, 0.06, 0.01))

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=atoms,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temperature,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_people, n_people), 10, 2.0),
        thermostat=AndersenThermostat(5.0),
        loggers=Dict("coords" => CoordinateLogger(10, dims=2),
                        "SIR" => SIRLogger(10, [])),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end
