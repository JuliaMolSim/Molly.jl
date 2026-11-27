@testset "Tiled GPUNeighborFinder kernels" begin
    # The tiled kernels of GPUNeighborFinder are KernelAbstractions/KernelInterface kernels,
    #   so they run on every GPU backend that supports them and also with `Array`s on the
    #   CPU backend of KernelAbstractions when it supports sub-groups of 32 work-items
    tiled_array_types = filter(AT -> Molly.supports_tiled_kernels(Molly.array_type_backend(AT)),
                               array_list)
    # The kernels for 64-wide sub-groups, which process a tile with each half of a sub-group,
    #   are also valid with 32-wide sub-groups, so they are tested on GPUs along with the
    #   kernels for the sub-group width of the backend (0); not on the CPU backend, where
    #   compiling the kernels for a system takes minutes
    sub_group_widths(AT) = (AT <: AbstractGPUArray ? (0, 2 * Molly.TILE_WIDTH) : (0,))

    # The force and energy kernels do not check for tiles marked `TILE_DEAD` by
    #   `prune_interacting_tiles_kernel!`, which `compact_interacting_tiles!` removes from
    #   the list. The list in the buffers is the one the last kernel was launched over.
    function test_no_dead_tiles(buffers)
        types = from_device(buffers.interacting_tiles_type)[1:buffers.num_pairs]
        @test !any(==(Molly.TILE_DEAD), types)
    end

    # The kernels add the virial to slots, which `gpu_forces!` folds, and the energy to
    #   `n_energy_slots` entries
    function tiled_forces_pe(sys)
        inters = Tuple(filter(Molly.use_neighbors, values(sys.pairwise_inters)))
        buffers = Molly.init_buffers_gpu(sys, 1)
        T, TH = Molly.float_type(sys), Molly.float_type_high(sys)
        fill!(buffers.fs_mat, zero(T))
        fill!(buffers.fs_mat_reordered, zero(T))
        fill!(buffers.virial_nounits, zero(TH))
        fill!(buffers.virial_slots, zero(TH))
        Molly.tiled_pairwise_forces!(buffers, sys, inters, Val(true), 0)
        test_no_dead_tiles(buffers)
        Molly.fold_virial_slots!(buffers.virial_nounits, buffers.virial_slots)
        fs_mat = from_device(buffers.fs_mat)
        fs = [SVector{size(fs_mat, 1)}(fs_mat[:, i]) for i in axes(fs_mat, 2)]
        pe_vec = KernelAbstractions.zeros(get_backend(sys.coords), TH, Molly.n_energy_slots)
        Molly.tiled_pairwise_pe!(pe_vec, buffers, sys, inters, 0)
        test_no_dead_tiles(buffers)
        return fs, from_device(buffers.virial_nounits), sum(from_device(pe_vec))
    end

    # The kernels are tested with the atoms ordered along both space-filling curves
    curves = (:morton, :hilbert)

    for AT in tiled_array_types, width in sub_group_widths(AT), (n_atoms, boundary) in (
                (20, CubicBoundary(3.0)),
                (1000, CubicBoundary(3.0)),
                (900, TriclinicBoundary(SVector(3.0, 0.0, 0.0), SVector(0.4, 3.0, 0.0),
                                        SVector(0.3, 0.5, 3.0))),
            ), curve in curves
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = width
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
                                special_pairs=special_pairs, curve=curve, array_type=AT),
                           force_units=NoUnits, energy_units=NoUnits)

        fs_ref, vir_ref = forces_virial(cpu_sys)
        pe_ref = potential_energy(cpu_sys)
        fs, vir, pe = tiled_forces_pe(tiled_sys)
        @test maximum(norm.(fs .- fs_ref)) < 1e-10 * maximum(norm.(fs_ref))
        @test isapprox(vir, vir_ref; rtol=1e-10)
        @test isapprox(pe, pe_ref; rtol=1e-10)
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = 0
    end

    # Several steps with the same buffers and a buffer distance, so that the displacement
    #   check made with the reorder, the inner masks of the dynamic pruning and a rebuild
    #   after a check that fails are used, in 3D for every array type and in 2D for GPU arrays
    function tiled_forces_pe!(buffers, sys, step_n)
        inters = Tuple(filter(Molly.use_neighbors, values(sys.pairwise_inters)))
        T, TH = Molly.float_type(sys), Molly.float_type_high(sys)
        fill!(buffers.fs_mat, zero(T))
        fill!(buffers.fs_mat_reordered, zero(T))
        fill!(buffers.virial_nounits, zero(TH))
        fill!(buffers.virial_slots, zero(TH))
        Molly.tiled_pairwise_forces!(buffers, sys, inters, Val(true), step_n)
        test_no_dead_tiles(buffers)
        Molly.fold_virial_slots!(buffers.virial_nounits, buffers.virial_slots)
        fs_mat = from_device(buffers.fs_mat)
        fs = [SVector{size(fs_mat, 1)}(fs_mat[:, i]) for i in axes(fs_mat, 2)]
        fill!(buffers.pe_vec_nounits, zero(TH))
        Molly.tiled_pairwise_pe!(buffers.pe_vec_nounits, buffers, sys, inters, step_n)
        test_no_dead_tiles(buffers)
        return fs, from_device(buffers.virial_nounits), sum(from_device(buffers.pe_vec_nounits))
    end

    for AT in tiled_array_types, width in sub_group_widths(AT), D in (3, 2), curve in curves
        (D == 2 && AT == Array) && continue
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = width
        Random.seed!(D)
        n_atoms = 1000
        r_cut = 1.0
        boundary = (D == 3 ? CubicBoundary(3.0) : RectangularBoundary(6.0))
        coords = place_atoms(n_atoms, boundary; min_dist=(D == 3 ? 0.1 : 0.12))
        atoms = [Atom(index=i, mass=10.0, charge=(isodd(i) ? 0.2 : -0.2), σ=0.2, ϵ=0.2)
                 for i in 1:n_atoms]
        pairwise_inters = (LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut),
                                        weight_special=0.5),
                           CoulombReactionField(dist_cutoff=r_cut, use_neighbors=true,
                                                weight_special=0.5, coulomb_const=138.935458))
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
                                dist_cutoff=(r_cut + 0.3), excluded_pairs=excluded_pairs,
                                special_pairs=special_pairs, curve=curve, array_type=AT),
                           force_units=NoUnits, energy_units=NoUnits)
        nf = tiled_sys.neighbor_finder
        buffers = Molly.init_buffers_gpu(tiled_sys, 1)
        rng = Xoshiro(D)
        for step_n in 0:4
            if 1 <= step_n <= 3
                # Small moves, which the tile list and the inner masks allow
                shifts = [SVector{D}(0.005 .* randn(rng, D)) for _ in 1:n_atoms]
            elseif step_n == 4
                # Two atoms moving 0.2 nm each, more than the buffer distance between them
                shifts = [(i <= 2 ? SVector{D}(ntuple(k -> k == 1 ? 0.2 : 0.0, D)) :
                                    zero(SVector{D, Float64})) for i in 1:n_atoms]
            end
            if step_n >= 1
                cpu_sys.coords .= wrap_coords.(cpu_sys.coords .+ shifts, (boundary,))
                tiled_sys.coords .= to_device(cpu_sys.coords, AT)
            end
            fs_ref, vir_ref = forces_virial(cpu_sys)
            pe_ref = potential_energy(cpu_sys)
            fs, vir, pe = tiled_forces_pe!(buffers, tiled_sys, step_n)
            @test maximum(norm.(fs .- fs_ref)) < 1e-10 * maximum(norm.(fs_ref))
            @test isapprox(vir, vir_ref; rtol=1e-10)
            @test isapprox(pe, pe_ref; rtol=1e-10)
            counts = Molly.neighbor_update_counts(nf)
            if step_n == 3
                @test counts.n_builds == 1
                @test counts.n_checks == 7
            elseif step_n == 4
                @test counts.n_builds == 2
            end
            if D == 2 && step_n in (0, 4)
                # The pruning found tiles without an in-range pair after the first build and
                #   the rebuild, so the lists tested above did hold dead tiles before the
                #   compaction. Not in 3D, where the box is so small that every pair of
                #   blocks holds an in-range pair.
                @test buffers.num_pairs < buffers.num_tile_candidates
            end
        end
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = 0
    end

    # A Float32 system on every GPU array type, including those without Float64 such as Metal
    #   arrays, with both sub-group widths of the kernels
    tiled_gpu_array_types_f32 = filter(AT -> AT <: AbstractGPUArray &&
                            Molly.supports_tiled_kernels(Molly.array_type_backend(AT)),
                        array_list_metal)
    for AT in tiled_gpu_array_types_f32, width in sub_group_widths(AT), curve in curves
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = width
        Random.seed!(32)
        n_atoms = 1000
        r_cut = 1.0f0
        boundary = CubicBoundary(3.0f0)
        coords = place_atoms(n_atoms, boundary; min_dist=0.1f0)
        atoms = [Atom(index=i, mass=10.0f0, charge=(isodd(i) ? 0.2f0 : -0.2f0), σ=0.2f0,
                      ϵ=0.2f0, λ=1.0f0) for i in 1:n_atoms]
        pairwise_inters = (LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut),
                                        weight_special=0.5f0),
                           CoulombReactionField(dist_cutoff=r_cut, use_neighbors=true,
                                                solvent_dielectric=78.3f0,
                                                weight_special=0.5f0,
                                                coulomb_const=138.935458f0))
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
                                special_pairs=special_pairs, curve=curve, array_type=AT),
                           force_units=NoUnits, energy_units=NoUnits)

        fs_ref, vir_ref = forces_virial(cpu_sys)
        pe_ref = potential_energy(cpu_sys)
        fs, vir, pe = tiled_forces_pe(tiled_sys)
        @test maximum(norm.(fs .- fs_ref)) < 1e-5 * maximum(norm.(fs_ref))
        @test isapprox(vir, vir_ref; rtol=1e-4)
        @test isapprox(pe, pe_ref; rtol=1e-4)
        Molly.TILE_SUB_GROUP_WIDTH_OVERRIDE[] = 0
    end

    # A Float32 system with the default Float64 `coulomb_const` and `solvent_dielectric` of
    #   CoulombReactionField, so that the pair energies are Float64 while the energy is
    #   accumulated in Float32: the accumulator must not become a `Union` that is only
    #   Float64 on some work-items, which gave a NaN energy on the CPU backend
    for AT in tiled_array_types, curve in curves
        Random.seed!(777)
        n_atoms = 777
        boundary = CubicBoundary(2.8f0u"nm")
        r_cut = 1.0f0u"nm"
        coords = place_atoms(n_atoms, boundary; min_dist=0.1f0u"nm")
        atoms = [Atom(index=i, mass=10.0f0u"g/mol", charge=(isodd(i) ? 0.3f0 : -0.3f0),
                      σ=0.2f0u"nm", ϵ=0.2f0u"kJ * mol^-1") for i in 1:n_atoms]
        pairwise_inters = (LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut),
                                        weight_special=0.5f0),
                           CoulombReactionField(dist_cutoff=r_cut, use_neighbors=true,
                                                weight_special=0.5f0))
        @test pairwise_inters[2].coulomb_const isa Quantity{Float64}
        sys_all = System(atoms=atoms, coords=coords, boundary=boundary,
                         pairwise_inters=pairwise_inters,
                         neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms,
                                                                dist_cutoff=r_cut))
        close_pairs = [(nb[1], nb[2]) for nb in find_neighbors(sys_all).list]
        shuffle!(close_pairs)
        excluded_pairs = close_pairs[1:900]
        special_pairs = close_pairs[901:1400]
        cpu_sys = System(atoms=atoms, coords=coords, boundary=boundary,
                         pairwise_inters=pairwise_inters,
                         neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms,
                            dist_cutoff=r_cut, excluded_pairs=excluded_pairs,
                            special_pairs=special_pairs))
        tiled_sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                           boundary=boundary, pairwise_inters=pairwise_inters,
                           neighbor_finder=GPUNeighborFinder(n_atoms=n_atoms,
                                dist_cutoff=r_cut, excluded_pairs=excluded_pairs,
                                special_pairs=special_pairs, curve=curve, array_type=AT))

        fs_ref = ustrip_vec.(forces(cpu_sys))
        pe_ref = ustrip(potential_energy(cpu_sys))
        fs, _, pe = tiled_forces_pe(tiled_sys)
        @test maximum(norm.(fs .- fs_ref)) < 1e-5 * maximum(norm.(fs_ref))
        @test isapprox(pe, pe_ref; rtol=1e-5)
    end

    # `compact_interacting_tiles!` removes the tiles marked `TILE_DEAD` by the pruning from
    #   the list and keeps the others in their order with their data, since the tree search
    #   sorts the list, and sets both tile counts, of the host and of the device
    for AT in tiled_array_types
        n_atoms = 1000
        boundary = CubicBoundary(3.0)
        coords = place_atoms(n_atoms, boundary; min_dist=0.1)
        atoms = [Atom(index=i, mass=10.0, charge=0.0, σ=0.2, ϵ=0.2) for i in 1:n_atoms]
        sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                     boundary=boundary,
                     pairwise_inters=(LennardJones(use_neighbors=true,
                                                   cutoff=DistanceCutoff(1.0)),),
                     neighbor_finder=GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=1.0,
                                                       array_type=AT),
                     force_units=NoUnits, energy_units=NoUnits)
        buffers = Molly.init_buffers_gpu(sys, 1)
        capacity = length(buffers.interacting_tiles_i)
        dead = Molly.TILE_DEAD
        type_lists = (
            UInt8[],                                    # empty list
            UInt8[0x00, 0x01, 0x00, 0x01],              # none dead
            fill(dead, 5),                              # all dead
            UInt8[dead],                                # a single tile, dead
            UInt8[0x01],                                # a single tile, live
            UInt8[dead, 0x00, 0x01, 0x00, dead],        # first and last dead
            UInt8[0x00, dead, dead, dead, 0x01, dead, 0x00, dead, 0x01, 0x00],
            rand(Xoshiro(5), (0x00, 0x01, dead), 500),  # more than one work-group
        )
        for types in type_lists
            n = length(types)
            # Different values for every tile and field, so that mixing up tiles or
            #   fields shows, and other values past the end of the list
            tiles_i = Int32.(1:capacity)
            tiles_j = Int32.(10_000 .+ (1:capacity))
            tiles_diag = UInt32.(0x00100000 .+ (1:capacity))
            tiles_type = fill(0x00, capacity)
            tiles_type[1:n] = types
            copyto!(buffers.interacting_tiles_i, to_device(tiles_i, AT))
            copyto!(buffers.interacting_tiles_j, to_device(tiles_j, AT))
            copyto!(buffers.interacting_tiles_type, to_device(tiles_type, AT))
            copyto!(buffers.interacting_tiles_diag, to_device(tiles_diag, AT))
            copyto!(buffers.num_interacting_tiles, to_device(Int32[n], AT))
            buffers.num_pairs = n

            Molly.compact_interacting_tiles!(buffers)

            keep = findall(!=(dead), types)
            n_live = length(keep)
            @test buffers.num_pairs == n_live
            @test only(from_device(buffers.num_interacting_tiles)) == n_live
            @test from_device(buffers.interacting_tiles_i)[1:n_live] == tiles_i[keep]
            @test from_device(buffers.interacting_tiles_j)[1:n_live] == tiles_j[keep]
            @test from_device(buffers.interacting_tiles_type)[1:n_live] == types[keep]
            @test from_device(buffers.interacting_tiles_diag)[1:n_live] == tiles_diag[keep]
        end
    end
end
