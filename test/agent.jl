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

    # Custom PairwiseInteraction
    struct SIRInteraction <: PairwiseInteraction
        nl_only::Bool
        dist_infection::Float64
        prob_infection::Float64
        prob_recovery::Float64
    end

    # Custom Logger
    struct SIRLogger
        n_steps::Int
        fracs_sir::Vector{Vector{Float64}}
    end

    # Custom force function
    function Molly.force(inter::SIRInteraction, dr, coord_i, coord_j, atom_i, atom_j, boundary)
        if (atom_i.status == infected && atom_j.status == susceptible) ||
                    (atom_i.status == susceptible && atom_j.status == infected)
            # Infect close people randomly
            r2 = sum(abs2, dr)
            if r2 < inter.dist_infection ^ 2 && rand() < inter.prob_infection
                atom_i.status = infected
                atom_j.status = infected
            end
        end
        # Workaround to obtain a self-interaction
        if atom_i.i == (atom_j.i + 1)
            # Recover randomly
            if atom_i.status == infected && rand() < inter.prob_recovery
                atom_i.status = recovered
            end
        end
        return zero(coord_i)
    end

    # Custom logging function
    function Molly.log_property!(logger::SIRLogger, s, neighbors, step_n; n_threads=Threads.nthreads(), kwargs...)
        if step_n % logger.n_steps == 0
            counts_sir = [
                count(p -> p.status == susceptible, s.atoms),
                count(p -> p.status == infected   , s.atoms),
                count(p -> p.status == recovered  , s.atoms)
            ]
            push!(logger.fracs_sir, counts_sir ./ length(s))
        end
    end

    n_people = 500
    n_steps = 1_000
    boundary = RectangularBoundary(10.0)
    temp = 1.0
    n_starting = 2
    atoms = [Person(i, i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]
    coords = place_atoms(n_people, boundary; min_dist=0.1)
    velocities = [velocity(1.0, temp; dims=2) for i in 1:n_people]
    pairwise_inters = (
        LennardJones=LennardJones(nl_only=true),
        SIR=SIRInteraction(false, 0.5, 0.06, 0.01),
    )
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=trues(n_people, n_people),
        n_steps=10,
        dist_cutoff=2.0,
    )
    simulator = VelocityVerlet(dt=0.02, coupling=AndersenThermostat(temp, 5.0))

    s = System(
        atoms=atoms,
        pairwise_inters=pairwise_inters,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinateLogger(Float64, 10; dims=2),
            SIR=SIRLogger(10, []),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    @time simulate!(s, simulator, n_steps; n_threads=1)
end
