@testset "GPU tile lists" begin
    # Define a simple system
    n_atoms = 100
    atom_mass = 10.0u"g/mol"
    σ = 0.3u"nm"
    ϵ = 1.0u"kJ * mol^-1"
    boundary = CubicBoundary(10.0u"nm")
    
    Random.seed!(42)
    coords = [SVector(rand(), rand(), rand()) * 10.0u"nm" for _ in 1:n_atoms]
    velocities = [zero(SVector{3, typeof(1.0u"nm/ps")}) for _ in 1:n_atoms]
    atoms = [Atom(index=i, mass=atom_mass, charge=0.0, σ=σ, ϵ=ϵ) for i in 1:n_atoms]
    
    # Lennard-Jones interaction
    pairwise_inters = (LennardJones(
        cutoff=DistanceCutoff(3.0u"nm"),
        use_neighbors=false,
    ),)
    
    # System on CPU
    sys_cpu = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        neighbor_finder=Molly.NoNeighborFinder(n_atoms),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    # System on GPU
    sys_gpu = System(
        atoms=CuArray(atoms),
        coords=CuArray(coords),
        velocities=CuArray(velocities),
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        neighbor_finder=nothing,
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    f_cpu = forces(sys_cpu)
    f_gpu = forces(sys_gpu, nothing)
    f_gpu_host = Array(f_gpu)
    
    for i in 1:n_atoms
        @test f_cpu[i] ≈ f_gpu_host[i] atol=1e-4u"kJ * mol^-1 * nm^-1"
    end

    pe_cpu = potential_energy(sys_cpu)
    pe_gpu = potential_energy(sys_gpu, nothing)
    
    @test pe_cpu ≈ pe_gpu atol=1e-3u"kJ * mol^-1"
end
