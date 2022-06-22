@testset "Energy minimization" begin
    coords = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(1.6, 1.0, 1.0)u"nm",
        SVector(1.4, 1.6, 1.0)u"nm",
    ]
    sys = System(
        atoms=[Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3],
        pairwise_inters=(LennardJones(),),
        coords=coords,
        box_size=CubicBoundary(5.0u"nm", 5.0u"nm", 5.0u"nm"),
    )
    sim = SteepestDescentMinimizer(tol=1.0u"kJ * mol^-1 * nm^-1")

    simulate!(sys, sim)
    dists = distances(sys.coords, sys.box_size)
    dists_flat = dists[triu(trues(3, 3), 1)]
    @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
    @test isapprox(potential_energy(sys), -3.0u"kJ * mol^-1";
                    atol=1e-4u"kJ * mol^-1")

    # No units
    coords = [
        SVector(1.0, 1.0, 1.0),
        SVector(1.6, 1.0, 1.0),
        SVector(1.4, 1.6, 1.0),
    ]
    sys = System(
        atoms=[Atom(σ=0.4 / (2 ^ (1 / 6)), ϵ=1.0) for i in 1:3],
        pairwise_inters=(LennardJones(force_units=NoUnits, energy_units=NoUnits),),
        coords=coords,
        box_size=CubicBoundary(5.0, 5.0, 5.0),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    sim = SteepestDescentMinimizer(step_size=0.01, tol=1.0)

    simulate!(sys, sim)
    dists = distances(sys.coords, sys.box_size) * u"nm"
    dists_flat = dists[triu(trues(3, 3), 1)]
    @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
    @test isapprox(potential_energy(sys) * u"kJ * mol^-1", -3.0u"kJ * mol^-1";
                    atol=1e-4u"kJ * mol^-1")

    if run_gpu_tests
        coords = cu([
            SVector(1.0, 1.0, 1.0)u"nm",
            SVector(1.6, 1.0, 1.0)u"nm",
            SVector(1.4, 1.6, 1.0)u"nm",
        ])
        sys = System(
            atoms=cu([Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3]),
            pairwise_inters=(LennardJones(),),
            coords=coords,
            box_size=CubicBoundary(5.0u"nm", 5.0u"nm", 5.0u"nm"),
        )
        sim = SteepestDescentMinimizer(tol=1.0u"kJ * mol^-1 * nm^-1")
    
        simulate!(sys, sim)
        dists = distances(sys.coords, sys.box_size)
        dists_flat = dists[triu(trues(3, 3), 1)]
        @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
        neighbors = find_neighbors(sys)
        @test isapprox(potential_energy(sys, neighbors), -3.0u"kJ * mol^-1";
                        atol=1e-4u"kJ * mol^-1")
    end
end
