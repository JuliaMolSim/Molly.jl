@testset "Energy minimization" begin
    coords_start = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(1.6, 1.0, 1.0)u"nm",
        SVector(1.4, 1.6, 1.0)u"nm",
    ]
    tol = 0.1u"kJ * mol^-1 * nm^-1"
    minimizers = (
        SteepestDescentMinimizer(tol=tol),
        FIREMinimizer(tol=tol),
        LBFGSMinimizer(tol=tol),
    )

    for minimizer in minimizers
        sys = System(
            atoms=[Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3],
            coords=copy(coords_start),
            boundary=CubicBoundary(5.0u"nm"),
            pairwise_inters=(LennardJones(),),
        )

        simulate!(sys, minimizer)
        dists = distances(sys.coords, sys.boundary)
        dists_flat = dists[triu(trues(3, 3), 1)]
        @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
        @test isapprox(potential_energy(sys; n_threads=1), -3.0u"kJ * mol^-1";
                        atol=1e-4u"kJ * mol^-1")
        @test maximum(norm, forces(sys; n_threads=1)) < tol
        # The velocities should not be modified by minimization
        @test iszero(sys.velocities)
    end

    # No units
    coords_start_nounits = [
        SVector(1.0, 1.0, 1.0),
        SVector(1.6, 1.0, 1.0),
        SVector(1.4, 1.6, 1.0),
    ]
    minimizers_nounits = (
        SteepestDescentMinimizer(step_size=0.01, tol=0.1),
        FIREMinimizer(dt=0.001, dt_max=0.01, tol=0.1),
        LBFGSMinimizer(step_size=0.01, tol=0.1),
    )

    for minimizer in minimizers_nounits
        sys = System(
            atoms=[Atom(σ=0.4 / (2 ^ (1 / 6)), ϵ=1.0, mass=1.0) for i in 1:3],
            coords=copy(coords_start_nounits),
            boundary=CubicBoundary(5.0),
            pairwise_inters=(LennardJones(),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, minimizer)
        dists = distances(sys.coords, sys.boundary) * u"nm"
        dists_flat = dists[triu(trues(3, 3), 1)]
        @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
        @test isapprox(potential_energy(sys; n_threads=1) * u"kJ * mol^-1", -3.0u"kJ * mol^-1";
                        atol=1e-4u"kJ * mol^-1")
        @test maximum(norm, forces(sys; n_threads=1)) < 0.1
    end

    # Logging and error handling
    mk_sys() = System(
        atoms=[Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3],
        coords=copy(coords_start),
        boundary=CubicBoundary(5.0u"nm"),
        pairwise_inters=(LennardJones(),),
        loggers=(coords=CoordinatesLogger(1), energy=PotentialEnergyLogger(1)),
    )
    for minimizer in (SteepestDescentMinimizer(tol=tol, max_steps=5, log_stream=IOBuffer()),
                      FIREMinimizer(tol=tol, max_steps=5, log_stream=IOBuffer()),
                      LBFGSMinimizer(tol=tol, max_steps=5, log_stream=IOBuffer()))
        sys = mk_sys()
        simulate!(sys, minimizer; run_loggers=true)
        log_str = String(take!(minimizer.log_stream))
        @test occursin("potential energy", log_str)
        @test count("Step ", log_str) == 6
        @test length(values(sys.loggers.coords)) == 6
        @test length(values(sys.loggers.energy)) == 6
        @test values(sys.loggers.energy)[end] < values(sys.loggers.energy)[1]
    end

    @test_throws ArgumentError simulate!(mk_sys(), FIREMinimizer(alpha_start=1.5))
    @test_throws ArgumentError simulate!(mk_sys(), LBFGSMinimizer(n_history=0))
    @test_throws ArgumentError simulate!(mk_sys(), LBFGSMinimizer(max_line_search_steps=0))

    for AT in array_list[2:end]
        for minimizer in minimizers
            sys = System(
                atoms=to_device([Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1")
                                 for i in 1:3], AT),
                coords=to_device(copy(coords_start), AT),
                boundary=CubicBoundary(5.0u"nm"),
                pairwise_inters=(LennardJones(),),
            )

            simulate!(sys, minimizer)
            dists = from_device(distances(sys.coords, sys.boundary))
            dists_flat = dists[triu(trues(3, 3), 1)]
            @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-2u"nm"), dists_flat)
            @test isapprox(potential_energy(sys), -3.0u"kJ * mol^-1";
                            atol=1e-2u"kJ * mol^-1")
            @test maximum(norm, forces(sys)) < tol
        end
    end
end
