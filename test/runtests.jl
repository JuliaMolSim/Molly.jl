using Molly

using Base.Threads
using Statistics
using Test

if nthreads() == 1
    @warn "The parallel tests will not be run as Julia is running on 1 thread"
else
    @info "The parallel tests will be run as Julia is running on $(nthreads()) threads"
end

@testset "Spatial" begin
    @test vector1D(4.0, 6.0, 10.0) ==  2.0
    @test vector1D(1.0, 9.0, 10.0) == -2.0
    @test vector1D(6.0, 4.0, 10.0) == -2.0
    @test vector1D(9.0, 1.0, 10.0) ==  2.0

    @test vector(SVector(4.0, 1.0, 6.0),
                    SVector(6.0, 9.0, 4.0), 10.0) == SVector(2.0, -2.0, -2.0)

    @test adjust_bounds(8.0 , 10.0) == 8.0
    @test adjust_bounds(12.0, 10.0) == 2.0
    @test adjust_bounds(-2.0, 10.0) == 8.0

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=[Atom(), Atom(), Atom()],
        coords=[SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0),
                SVector(5.0, 5.0, 5.0)],
        box_size=10.0,
        neighbour_finder=DistanceNeighbourFinder(trues(3, 3), 10, 2.0)
    )
    find_neighbours!(s, s.neighbour_finder, 0, parallel=false)
    @test s.neighbours == [(2, 1)]
    if nthreads() > 1
        find_neighbours!(s, s.neighbour_finder, 0, parallel=true)
        @test s.neighbours == [(2, 1)]
    end
end

temp = 298
timestep = 0.002
n_steps = 20_000
box_size = 2.0

@testset "Lennard-Jones gas 2D" begin
    n_atoms = 20

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                    mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        general_inters=(LennardJones(true),),
        coords=[box_size .* rand(SVector{2}) for i in 1:n_atoms],
        velocities=[velocity(10.0, temp, dims=2) .* 0.01 for i in 1:n_atoms],
        temperature=temp,
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
        thermostat=AndersenThermostat(10.0),
        loggers=Dict("temp" => TemperatureLogger(100),
                     "coords" => CoordinateLogger(100, dims=2)),
        timestep=timestep,
        n_steps=n_steps
    )

    show(devnull, s)

    @time simulate!(s, parallel=false)
end

@testset "Lennard-Jones gas" begin
    n_atoms = 100
    parallel_list = nthreads() > 1 ? (false, true) : (false,)

    for parallel in parallel_list
        s = Simulation(
            simulator=VelocityVerlet(),
            atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                        mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
            general_inters=(LennardJones(true),),
            coords=[box_size .* rand(SVector{3}) for i in 1:n_atoms],
            velocities=[velocity(10.0, temp) .* 0.01 for i in 1:n_atoms],
            temperature=temp,
            box_size=box_size,
            neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),
            thermostat=AndersenThermostat(10.0),
            loggers=Dict("temp" => TemperatureLogger(100),
                         "coords" => CoordinateLogger(100),
                         "energy" => EnergyLogger(100)),
            timestep=timestep,
            n_steps=n_steps
        )

        @time simulate!(s, parallel=parallel)

        final_coords = last(s.loggers["coords"].coords)
        @test minimum(minimum.(final_coords)) > 0.0
        @test maximum(maximum.(final_coords)) < box_size
        displacements(final_coords, box_size)
        distances(final_coords, box_size)
        rdf(final_coords, box_size)
    end
end

@testset "Lennard-Jones gas energy conservation" begin
    temp = 1.
    timestep = 0.005
    n_steps = 10_000
    box_size = 50.0
    n_atoms = 2000
    m = 0.8 * box_size^3 / n_atoms

    parallel_list = nthreads() > 1 ? (false, true) : (false,)

    for parallel in parallel_list
        s = Simulation(
            simulator=VelocityVerlet(),
            atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                        mass=m, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
            general_inters=(LennardJones(),),
            coords=placeatoms(n_atoms, box_size, 0.6),
            velocities=[velocity(10.0, temp) .* 0.01 for i in 1:n_atoms],
            temperature=temp,
            box_size=box_size,
            loggers=Dict("coords" => CoordinateLogger(100),
                        "energy" => EnergyLogger(100)),
            timestep=timestep,
            n_steps=n_steps
        )

        E0 = energy(s)
        @time simulate!(s, parallel=true)

        ΔE = energy(s) - E0
        @test abs(ΔE) < 2e-2

        Es = s.loggers["energy"].energies
        maxΔE = maximum(abs.(Es .- E0))
        @test maxΔE < 2e-2

        @test abs(Es[end] - Es[1]) < 2e-2
        @test std(Es.-Es[1])/n_atoms < timestep^2

        final_coords = last(s.loggers["coords"].coords)
        @test minimum(minimum.(final_coords)) > 0.0
        @test maximum(maximum.(final_coords)) < box_size
    end
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
        temperature=temp,
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
        velocities=[velocity(10.0, temp) .* 0.01 for i in 1:n_atoms],
        temperature=temp,
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

    true_n_atoms = 5191
    @test length(atoms) == true_n_atoms
    @test length(coords) == true_n_atoms
    @test size(nb_matrix) == (true_n_atoms, true_n_atoms)
    @test length(specific_inter_lists) == 3
    @test length(general_inters) == 2
    @test box_size == 3.7146
    show(devnull, first(atoms))

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=[velocity(a.mass, temp) .* 0.01 for a in atoms],
        temperature=temp,
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
        velocities=[velocity(Float32, a.mass, temp) .* 0.01f0 for a in atoms],
        temperature=Float32(temp),
        box_size=box_size,
        neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10, 1.2f0),
        thermostat=AndersenThermostat(10.0f0),
        loggers=Dict("temp" => TemperatureLogger(Float32, 10),
                        "coords" => CoordinateLogger(Float32, 10)),
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
        return nothing
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

    temp = 0.01
    timestep = 0.02
    box_size = 10.0
    n_steps = 1_000
    n_people = 500
    n_starting = 2
    atoms = [Person(i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]
    coords = [box_size .* rand(SVector{2}) for i in 1:n_people]
    velocities = [velocity(1.0, temp, dims=2) for i in 1:n_people]
    general_inters = (LennardJones = LennardJones(true),
                            SIR = SIRInteraction(false, 0.5, 0.06, 0.01))

    s = Simulation(
        simulator=VelocityVerlet(),
        atoms=atoms,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temp,
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
