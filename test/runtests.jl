using Molly
using CUDA

using Base.Threads
using Statistics
using Test

@warn "This file does not include all the tests for Molly.jl due to CI time limits, " *
        "see the test directory for more"

if nthreads() > 1
    @info "The parallel tests will be run as Julia is running on $(nthreads()) threads"
else
    @warn "The parallel tests will not be run as Julia is running on 1 thread"
end

if CUDA.functional()
    @info "The GPU tests will be run as CUDA is available"
else
    @warn "The GPU tests will not be run as CUDA is not available"
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
    bonds = [HarmonicBond(i, i + (n_atoms ÷ 2), 0.1, 300_000.0) for i in 1:(n_atoms ÷ 2)]

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
        loggers=Dict("temp" => TemperatureLogger(10),
                        "coords" => CoordinateLogger(10),
                        "energy" => EnergyLogger(10)),
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
                        "coords" => CoordinateLogger(Float32, 10),
                        "energy" => EnergyLogger(Float32, 10)),
        timestep=timestep,
        n_steps=n_steps
    )

    @time simulate!(s, parallel=false)
end

@testset "General interactions" begin
    n_atoms = 100
    #general_inter_types = (LennardJones(true), SoftSphere(true), Mie(5, 10, true), Coulomb(true),
    general_inter_types = (LennardJones(true), SoftSphere(true), Coulomb(true),
                            Gravity(10.0, true))

    for gi in general_inter_types
        s = Simulation(
            simulator=VelocityVerlet(),
            atoms=[Atom(charge=i % 2 == 0 ? -1.0 : 1.0, mass=10.0, σ=0.2, ϵ=0.2) for i in 1:n_atoms],
            general_inters=(gi,),
            coords=placeatoms(n_atoms, box_size, 0.2),
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

        @time simulate!(s)
    end
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

@testset "Different implementations" begin
    function placediatomics(n_molecules, box_size, min_dist, bond_length)
        min_dist_sq = min_dist ^ 2
        coords = SArray{Tuple{3}, Float64, 1, 3}[]
        while length(coords) < (n_molecules * 2)
            new_coord_a = rand(SVector{3}) .* box_size
            new_coord_b = copy(new_coord_a) + SVector{3}([bond_length, 0.0, 0.0])
            okay = new_coord_b[1] <= box_size
            for coord in coords
                if sum(abs2, vector(coord, new_coord_a, box_size)) < min_dist_sq ||
                        sum(abs2, vector(coord, new_coord_b, box_size)) < min_dist_sq
                    okay = false
                    break
                end
            end
            if okay
                push!(coords, new_coord_a)
                push!(coords, new_coord_b)
            end
        end
        return coords
    end

    n_atoms = 400
    mass = 10.0
    box_size = 6.0
    temp = 1.0
    starting_coords = placediatomics(n_atoms ÷ 2, box_size, 0.2, 0.2)
    starting_velocities = [velocity(mass, temp) for i in 1:n_atoms]
    starting_coords_f32 = [Float32.(c) for c in starting_coords]
    starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

    function runsim(nl::Bool, parallel::Bool, gpu_diff_safe::Bool, f32::Bool, gpu::Bool)
        n_atoms = 400
        n_steps = 200
        mass = f32 ? 10.0f0 : 10.0
        box_size = f32 ? 6.0f0 : 6.0
        timestep = f32 ? 0.02f0 : 0.02
        temp = f32 ? 1.0f0 : 1.0
        simulator = VelocityVerlet()
        thermostat = NoThermostat()
        bonds = [HarmonicBond((i * 2) - 1, i * 2, f32 ? 0.2f0 : 0.2, f32 ? 10_000.0f0 : 10_000.0) for i in 1:(n_atoms ÷ 2)]
        specific_inter_lists = (bonds,)

        neighbour_finder = NoNeighbourFinder()
        general_inters = (LennardJones(false),)
        if nl
            neighbour_finder = DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, f32 ? 1.5f0 : 1.5)
            general_inters = (LennardJones(true),)
        end

        if gpu
            coords = cu(deepcopy(f32 ? starting_coords_f32 : starting_coords))
            velocities = cu(deepcopy(f32 ? starting_velocities_f32 : starting_velocities))
            atoms = cu([AtomMin(f32 ? 0.0f0 : 0.0, mass, f32 ? 0.2f0 : 0.2, f32 ? 0.2f0 : 0.2) for i in 1:n_atoms])
        else
            coords = deepcopy(f32 ? starting_coords_f32 : starting_coords)
            velocities = deepcopy(f32 ? starting_velocities_f32 : starting_velocities)
            atoms = [Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=f32 ? 0.0f0 : 0.0,
                            mass=mass, σ=f32 ? 0.2f0 : 0.2, ϵ=f32 ? 0.2f0 : 0.2) for i in 1:n_atoms]
        end

        s = Simulation(
            simulator=simulator,
            atoms=atoms,
            specific_inter_lists=specific_inter_lists,
            general_inters=general_inters,
            coords=coords,
            velocities=velocities,
            temperature=temp,
            box_size=box_size,
            neighbour_finder=neighbour_finder,
            thermostat=thermostat,
            timestep=timestep,
            n_steps=n_steps,
            gpu_diff_safe=gpu_diff_safe
        )

        c = simulate!(s, parallel=parallel)
        return c
    end

    runs = [
        ("in-place"        , [false, false, false, false, false]),
        ("in-place NL"     , [true , false, false, false, false]),
        ("in-place f32"    , [false, false, false, true , false]),
        ("out-of-place"    , [false, false, true , false, false]),
        ("out-of-place f32", [false, false, true , true , false]),
    ]
    if nthreads() > 1
        push!(runs, ("in-place parallel"   , [false, true , false, false, false]))
        push!(runs, ("in-place NL parallel", [true , true , false, false, false]))
    end
    if CUDA.functional()
        push!(runs, ("out-of-place gpu"    , [false, false, true , false, true ]))
        push!(runs, ("out-of-place gpu f32", [false, false, true , true , true ]))
    end

    final_coords_ref = runsim(runs[1][2]...)
    for (name, args) in runs
        final_coords = runsim(args...)
        final_coords_f64 = [Float64.(c) for c in final_coords]
        diff = sum(sum(map(x -> abs.(x), final_coords_f64 .- final_coords_ref))) / (3 * n_atoms)
        # Check all simulations give the same result to within some error
        @info "$(rpad(name, 20)) - difference per coordinate $diff"
        @test diff < 1e-4
    end
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
