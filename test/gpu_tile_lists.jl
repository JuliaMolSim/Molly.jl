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
    pairwise_inters_cpu = (LennardJones(
        cutoff=DistanceCutoff(3.0u"nm"),
        use_neighbors=false,
    ),)

    pairwise_inters_gpu = (LennardJones(
        cutoff=DistanceCutoff(3.0u"nm"),
        use_neighbors=true,
    ),)
    
    # System on CPU
    sys_cpu = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=pairwise_inters_cpu,
        neighbor_finder=Molly.NoNeighborFinder(),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    # System on GPU
    sys_gpu = System(
        atoms=CuArray(atoms),
        coords=CuArray(coords),
        velocities=CuArray(velocities),
        boundary=boundary,
        pairwise_inters=pairwise_inters_gpu,
        neighbor_finder=Molly.GPUNeighborFinder(
            n_atoms=n_atoms,
            dist_cutoff=3.0u"nm",
            device_vector_type=CuArray{Int32, 1},
        ),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    f_cpu = forces(sys_cpu)
    f_gpu = forces(sys_gpu, nothing)
    f_gpu_host = Array(f_gpu)
    
    for i in 1:n_atoms
        @test isapprox(f_cpu[i], f_gpu_host[i], atol=1e-10u"kJ * mol^-1 * nm^-1")
    end

    pe_cpu = potential_energy(sys_cpu)
    pe_gpu = potential_energy(sys_gpu, nothing)
    
    @test isapprox(pe_cpu, pe_gpu, atol=1e-10u"kJ * mol^-1")

    pairwise_inters_gpu_overflow = (LennardJones(
        cutoff=DistanceCutoff(20.0u"nm"),
        use_neighbors=true,
    ),)

    sys_gpu_overflow = System(
        atoms=CuArray(atoms),
        coords=CuArray(coords),
        velocities=CuArray(velocities),
        boundary=boundary,
        pairwise_inters=pairwise_inters_gpu_overflow,
        neighbor_finder=Molly.GPUNeighborFinder(
            n_atoms=n_atoms,
            dist_cutoff=20.0u"nm",
            device_vector_type=CuArray{Int32, 1},
        ),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )

    function with_tiny_tile_capacity(buffers)
        tiny_capacity = 1
        return Molly.BuffersGPU(
            buffers.fs_mat,
            buffers.pe_vec_nounits,
            buffers.virial,
            buffers.virial_nounits,
            buffers.kin_tensor,
            buffers.pres_tensor,
            buffers.box_mins,
            buffers.box_maxs,
            buffers.morton_seq,
            buffers.morton_seq_buffer_1,
            buffers.morton_seq_buffer_2,
            buffers.morton_seq_inv,
            buffers.compressed_masks,
            buffers.tile_is_clean,
            CUDA.zeros(Int32, tiny_capacity), # interacting_tiles_i
            CUDA.zeros(Int32, tiny_capacity), # interacting_tiles_j
            CUDA.zeros(UInt8, tiny_capacity), # interacting_tiles_type
            CUDA.zeros(Int32, 1),             # num_interacting_tiles (atomic counter)
            CUDA.zeros(Int32, 1),             # interacting_tiles_overflow
            buffers.coords_reordered,
            buffers.velocities_reordered,
            buffers.atoms_reordered,
            buffers.fs_mat_reordered,
            -1,
            buffers.last_r_cut,
            0, # num_pairs
        )
    end

    overflow_force_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1))
    overflow_energy_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1, true))

    @test_throws ErrorException Molly.forces!(
        Molly.zero_forces(sys_gpu_overflow),
        sys_gpu_overflow,
        nothing,
        overflow_force_buffers,
        Val(false),
        0,
    )
    @test_throws ErrorException potential_energy(sys_gpu_overflow, nothing, overflow_energy_buffers, 0)
end
