@testset "Agent-based modelling" begin
    @enum Status susceptible infected recovered

    # Custom atom type
    mutable struct Person
        i::Int
        status::Status
        mass::Float64
        σ::Float64
        ϵ::Float64
    end

    # Custom pairwise interaction
    struct SIRInteraction <: PairwiseInteraction
        dist_infection::Float64
        prob_infection::Float64
        prob_recovery::Float64
    end

    # Custom logger
    struct SIRLogger
        n_steps::Int
        fracs_sir::Vector{Vector{Float64}}
    end

    Base.values(logger::SIRLogger) = logger.fracs_sir

    # Custom force function
    function Molly.force(inter::SIRInteraction,
                            vec_ij,
                            atom_i,
                            atom_j,
                            args...)
        if (atom_i.status == infected && atom_j.status == susceptible) ||
                    (atom_i.status == susceptible && atom_j.status == infected)
            # Infect close people randomly
            r2 = sum(abs2, vec_ij)
            if r2 < inter.dist_infection^2 && rand() < inter.prob_infection
                atom_i.status = infected
                atom_j.status = infected
            end
        end
        # Workaround to obtain a self-interaction
        if atom_i.i == (atom_j.i - 1)
            # Recover randomly
            if atom_i.status == infected && rand() < inter.prob_recovery
                atom_i.status = recovered
            end
        end
        return zero(vec_ij)
    end

    # Test log_property! definition rather than just using GeneralObservableLogger
    function Molly.log_property!(logger::SIRLogger, sys, buffers, neighbors, step_n; kwargs...)
        if step_n % logger.n_steps == 0
            counts_sir = [
                count(p -> p.status == susceptible, sys.atoms),
                count(p -> p.status == infected   , sys.atoms),
                count(p -> p.status == recovered  , sys.atoms)
            ]
            push!(logger.fracs_sir, counts_sir ./ length(sys))
        end
    end

    rng = Xoshiro(15)
    temp = 1.0
    boundary = RectangularBoundary(10.0)
    n_steps = 1_000
    n_people = 500
    n_starting = 2
    atoms = [Person(i, i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]
    coords = place_atoms(n_people, boundary; min_dist=0.1, rng=rng)
    velocities = [random_velocity(1.0, temp; dims=2, rng=rng) for i in 1:n_people]

    lj = LennardJones(cutoff=DistanceCutoff(1.6), use_neighbors=true)
    sir = SIRInteraction(0.5, 0.06, 0.01)
    @test !use_neighbors(sir)
    pairwise_inters = (LennardJones=lj, SIR=sir)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_people, n_people),
        n_steps=10,
        dist_cutoff=2.0,
    )
    simulator = VelocityVerlet(
        dt=0.02,
        coupling=(AndersenThermostat(temp, 5.0),),
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinatesLogger(Float64, 10; dims=2),
            SIR=SIRLogger(10, []),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    @time simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

    s, i, r = values(sys.loggers.SIR)[end]
    @test s < 0.9
    @test i < 0.9
    @test r > 0.1
end
