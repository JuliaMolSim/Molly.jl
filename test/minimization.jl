@testset "Energy minimization" begin
    coords = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(1.6, 1.0, 1.0)u"nm",
        SVector(1.4, 1.6, 1.0)u"nm",
    ]
    sys = System(
        atoms=[Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3],
        coords=coords,
        boundary=CubicBoundary(5.0u"nm"),
        pairwise_inters=(LennardJones(),),
    )
    sim = SteepestDescentMinimizer(tol=1.0u"kJ * mol^-1 * nm^-1")

    simulate!(sys, sim)
    dists = distances(sys.coords, sys.boundary)
    dists_flat = dists[triu(trues(3, 3), 1)]
    @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
    @test isapprox(potential_energy(sys; n_threads=1), -3.0u"kJ * mol^-1";
                    atol=1e-4u"kJ * mol^-1")

    # No units
    coords = [
        SVector(1.0, 1.0, 1.0),
        SVector(1.6, 1.0, 1.0),
        SVector(1.4, 1.6, 1.0),
    ]
    sys = System(
        atoms=[Atom(σ=0.4 / (2 ^ (1 / 6)), ϵ=1.0, mass=1.0) for i in 1:3],
        coords=coords,
        boundary=CubicBoundary(5.0),
        pairwise_inters=(LennardJones(),),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    sim = SteepestDescentMinimizer(step_size=0.01, tol=1.0)

    simulate!(sys, sim)
    dists = distances(sys.coords, sys.boundary) * u"nm"
    dists_flat = dists[triu(trues(3, 3), 1)]
    @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-3u"nm"), dists_flat)
    @test isapprox(potential_energy(sys; n_threads=1) * u"kJ * mol^-1", -3.0u"kJ * mol^-1";
                    atol=1e-4u"kJ * mol^-1")

    for array_type in array_list[2:end]
        coords = array_type([
            SVector(1.0, 1.0, 1.0)u"nm",
            SVector(1.6, 1.0, 1.0)u"nm",
            SVector(1.4, 1.6, 1.0)u"nm",
        ])
        sys = System(
            atoms=array_type([Atom(σ=(0.4 / (2 ^ (1 / 6)))u"nm", ϵ=1.0u"kJ * mol^-1") for i in 1:3]),
            coords=coords,
            boundary=CubicBoundary(5.0u"nm"),
            pairwise_inters=(LennardJones(),),
        )
        sim = SteepestDescentMinimizer(tol=1.0u"kJ * mol^-1 * nm^-1")

        simulate!(sys, sim)
        dists = Array(distances(sys.coords, sys.boundary))
        dists_flat = dists[triu(trues(3, 3), 1)]

        # GPU tolerances are more lenient (possibly for f32 shenanigans)
        @test all(x -> isapprox(x, 0.4u"nm"; atol=1e-2u"nm"), dists_flat)
        @test isapprox(potential_energy(sys), -3.0u"kJ * mol^-1";
                        atol=1e-2u"kJ * mol^-1")
    end
end
