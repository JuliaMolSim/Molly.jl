using AtomsBase
using AtomsBaseTesting

# NEED METHODS TO ADD MISSING PIECES#

@testset "AbstractSystem -> Molly System" begin
    system = make_test_system().system
    test_approx_eq(system, System(system))
end

@testset "Molly System -> AbstractSystem -> Molly System" begin
    n_atoms = 100
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    s = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=coords,
        boundary=boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(coords=CoordinateLogger(100),),
    )

    #Conversion to AtomsBase type
    ps = periodic_system(s)

    @test test_approx_eq(s, ps)

    # Re-construct Molly System
    s2 = parse_system(ps)

    #Call update constructor
    s2 = System(s2; 
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(coords=CoordinateLogger(100),)
    )

end
