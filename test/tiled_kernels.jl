@testset "Tiled GPUNeighborFinder kernels" begin
    # The tiled kernels of GPUNeighborFinder are KernelAbstractions/KernelInterface kernels,
    #   so they also run with `Array`s on the CPU backend of KernelAbstractions when it
    #   supports sub-groups of 32 work-items
    tiled_array_types = filter(AT -> Molly.supports_tiled_kernels(get_backend(AT{Float32}(undef, 0))),
                               array_list)

    function tiled_forces_pe(sys)
        inters = Tuple(filter(Molly.use_neighbors, values(sys.pairwise_inters)))
        buffers = Molly.init_buffers_gpu(sys, 1)
        T, TH = Molly.float_type(sys), Molly.float_type_high(sys)
        fill!(buffers.fs_mat, zero(T))
        fill!(buffers.fs_mat_reordered, zero(T))
        fill!(buffers.virial_nounits, zero(TH))
        Molly.tiled_pairwise_forces!(buffers, sys, inters, Val(true), 0)
        fs_mat = from_device(buffers.fs_mat)
        fs = [SVector{size(fs_mat, 1)}(fs_mat[:, i]) for i in axes(fs_mat, 2)]
        pe_vec = KernelAbstractions.zeros(get_backend(sys.coords), TH, 1)
        Molly.tiled_pairwise_pe!(pe_vec, buffers, sys, inters, 0)
        return fs, from_device(buffers.virial_nounits), only(from_device(pe_vec))
    end

    for AT in tiled_array_types, (n_atoms, boundary) in (
                (20, CubicBoundary(3.0)),
                (1000, CubicBoundary(3.0)),
                (900, TriclinicBoundary(SVector(3.0, 0.0, 0.0), SVector(0.4, 3.0, 0.0),
                                        SVector(0.3, 0.5, 3.0))),
            )
        Random.seed!(n_atoms)
        r_cut = 1.0
        coords = place_atoms(n_atoms, boundary; min_dist=0.1)
        atoms = [Atom(index=i, mass=10.0, charge=0.0, σ=0.2, ϵ=0.2) for i in 1:n_atoms]
        pairwise_inters = (LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut),
                                        weight_special=0.5),)
        sys_all = System(atoms=atoms, coords=coords, boundary=boundary,
                         pairwise_inters=pairwise_inters,
                         neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms,
                                                                dist_cutoff=r_cut),
                         force_units=NoUnits, energy_units=NoUnits)
        close_pairs = [(nb[1], nb[2]) for nb in find_neighbors(sys_all).list]
        shuffle!(close_pairs)
        n_exceptions = length(close_pairs) ÷ 4
        excluded_pairs = close_pairs[1:n_exceptions]
        special_pairs = close_pairs[(n_exceptions + 1):(2 * n_exceptions)]

        cpu_sys = System(atoms=atoms, coords=coords, boundary=boundary,
                         pairwise_inters=pairwise_inters,
                         neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms,
                            dist_cutoff=r_cut, excluded_pairs=excluded_pairs,
                            special_pairs=special_pairs),
                         force_units=NoUnits, energy_units=NoUnits)
        tiled_sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                           boundary=boundary, pairwise_inters=pairwise_inters,
                           neighbor_finder=GPUNeighborFinder(n_atoms=n_atoms,
                                dist_cutoff=r_cut, excluded_pairs=excluded_pairs,
                                special_pairs=special_pairs, array_type=AT),
                           force_units=NoUnits, energy_units=NoUnits)

        fs_ref, vir_ref = forces_virial(cpu_sys)
        pe_ref = potential_energy(cpu_sys)
        fs, vir, pe = tiled_forces_pe(tiled_sys)
        @test maximum(norm.(fs .- fs_ref)) < 1e-10 * maximum(norm.(fs_ref))
        @test isapprox(vir, vir_ref; rtol=1e-10)
        @test isapprox(pe, pe_ref; rtol=1e-10)
    end
end
